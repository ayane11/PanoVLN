"""Native Pano HF backend; GPU parity must pass before use for training.

Sampling is untruncated categorical sampling at the configured temperature.
Both stored rollout probabilities and actor probabilities use that temperature.
No standard-Qwen fallback is permitted.
"""
from contextlib import nullcontext
from functools import lru_cache
import base64
from dataclasses import asdict
import io
import math
import torch
from .precision import kernel_construction
from PIL import Image
from transformers import AutoProcessor
from src.qwen_vl import Qwen3_5ForConditionalGenerationForPanoVLN
from src.eval.action_policy import build_action_token_lookup, extract_action_uncertainties
from .episode import Generation
from .inputs import build_inputs, ObservationConfig


def decode_frames(frames):
    return [Image.open(io.BytesIO(base64.b64decode(frame))).convert('RGB') for frame in frames]


def entropy_from_logits(logits):
    return logits.logsumexp(-1)-(logits.softmax(-1)*logits).sum(-1)


@lru_cache(maxsize=1)
def compiled_entropy():
    return torch.compile(entropy_from_logits,dynamic=True)


def compute_entropy(model,logits):
    fn=compiled_entropy() if getattr(model,'_dapo_compile_entropy',False) else entropy_from_logits
    return fn(logits)


def response_logprobs(model, prompt_inputs, response_ids, temperature=1.0, *, return_entropy=False):
    """Differentiable response likelihood; caller chooses grad/no_grad context."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Training temperature must be finite and positive')
    device = prompt_inputs['input_ids'].device
    response = torch.as_tensor(response_ids, dtype=torch.long, device=device).reshape(1,-1)
    if response.numel() == 0:
        raise ValueError('Cannot train an empty response')
    inputs = dict(prompt_inputs)
    inputs['input_ids'] = torch.cat([inputs['input_ids'], response], dim=-1)
    inputs['attention_mask'] = torch.cat([inputs['attention_mask'], torch.ones_like(response)], dim=-1)
    if 'mm_token_type_ids' in inputs:
        inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(response)], dim=-1)
    if 'position_ids' in inputs:
        raise ValueError('Explicit prompt-only positions require response extension')
    precision=getattr(model,'_dapo_compute_dtype',None)
    context=torch.autocast('cuda',dtype=torch.bfloat16) if precision==torch.bfloat16 else nullcontext()
    geometry=inputs.get('image_erp_geometry')
    hook=None
    if geometry is not None:
        expected=geometry.detach().clone()
        def check_geometry(module,args,kwargs):
            actual=kwargs['image_erp_geometry']
            if actual.dtype!=torch.float32 or not torch.equal(actual,expected):
                raise ValueError('FSDP changed ERP geometry precision/values')
        hook=getattr(model,'module',model).register_forward_pre_hook(check_geometry,with_kwargs=True)
    try:
        with context:
            output = model(**inputs, use_cache=False, logits_to_keep=response.shape[1]+1, return_dict=True)
    finally:
        if hook is not None: hook.remove()
    logits = output.logits[:, -response.shape[1]-1:-1, :].float() / temperature
    normalizer=logits.logsumexp(-1)
    logp=(logits.gather(-1,response.unsqueeze(-1)).squeeze(-1)-normalizer).squeeze(0)
    if return_entropy:
        entropy=compute_entropy(model,logits).squeeze(0)
        return logp,entropy
    return logp


def batch_response_logprobs(model, prompts, responses, temperatures, pad_token_id, *, return_entropy=False):
    """One padded forward, returning only each example's real response tokens."""
    from .batching import collate_inputs
    if not prompts or len(prompts)!=len(responses) or len(prompts)!=len(temperatures):
        raise ValueError('Mismatched/empty probability microbatch')
    rows=[]
    for prompt, ids, temperature in zip(prompts,responses,temperatures):
        if not math.isfinite(temperature) or temperature<=0:
            raise ValueError('Training temperature must be finite and positive')
        response=torch.as_tensor(ids,dtype=torch.long,device=prompt['input_ids'].device).reshape(1,-1)
        if not response.numel(): raise ValueError('Cannot train an empty response')
        row=dict(prompt)
        row['input_ids']=torch.cat([prompt['input_ids'],response],dim=-1)
        row['attention_mask']=torch.cat([prompt['attention_mask'],torch.ones_like(response)],dim=-1)
        if 'mm_token_type_ids' in row:
            row['mm_token_type_ids']=torch.cat([row['mm_token_type_ids'],torch.zeros_like(response)],dim=-1)
        rows.append(row)
    inputs=collate_inputs(rows,pad_token_id)
    width=max(map(len,responses))
    context=torch.autocast('cuda',dtype=torch.bfloat16) if getattr(model,'_dapo_compute_dtype',None)==torch.bfloat16 else nullcontext()
    geometry=inputs.get('image_erp_geometry');hook=None
    if geometry is not None:
        expected=geometry.detach().clone()
        def check_geometry(module,args,kwargs):
            actual=kwargs['image_erp_geometry']
            if actual.dtype!=torch.float32 or not torch.equal(actual,expected):
                raise ValueError('FSDP changed ERP geometry precision/values')
        hook=getattr(model,'module',model).register_forward_pre_hook(check_geometry,with_kwargs=True)
    try:
        with context:
            output=model(**inputs,use_cache=False,logits_to_keep=width+1,return_dict=True)
    finally:
        if hook is not None: hook.remove()
    values=[]
    for index,(ids,temperature) in enumerate(zip(responses,temperatures)):
        logits=output.logits[index,-len(ids)-1:-1,:].float()/temperature
        normalizer=logits.logsumexp(-1)
        target=torch.as_tensor(ids,dtype=torch.long,device=logits.device)
        logp=logits.gather(-1,target[:,None]).squeeze(-1)-normalizer
        if return_entropy:
            values.append((logp,compute_entropy(model,logits)))
        else: values.append(logp)
    return values


class NativePolicy:
    def __init__(self, model, processor, *, observation=ObservationConfig(), temperature=1.0,
                 max_response_tokens=256, uncertainty=False):
        if not math.isfinite(temperature) or temperature <= 0 or max_response_tokens < 1:
            raise ValueError('Invalid generation settings')
        self.model, self.processor = model, processor
        self.observation, self.temperature = observation, temperature
        self.max_response_tokens, self.uncertainty = max_response_tokens, uncertainty
        self.lookup = build_action_token_lookup(processor.tokenizer) if uncertainty else None

    @classmethod
    def load(cls, checkpoint, device='cuda:0', attention='flash_attention_2', dtype=torch.bfloat16, **kwargs):
        processor = AutoProcessor.from_pretrained(checkpoint,local_files_only=True)
        with kernel_construction(dtype,require_fla=False):
            model = Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(
                checkpoint,local_files_only=True,torch_dtype=dtype,attn_implementation=attention)
        model.to(device).eval()
        if 'observation' not in kwargs:
            kwargs['observation'] = ObservationConfig(
                top_crop_degrees=float(model.config.erp_top_crop_degrees),
                bottom_crop_degrees=float(model.config.erp_bottom_crop_degrees),
                panovggt_enabled=bool(model.config.panovggt_enabled))
        return cls(model,processor,**kwargs)

    def prepare(self, instruction, frames):
        result = build_inputs(self.processor,instruction,decode_frames(frames),self.observation)
        result['inputs'] = result['inputs'].to(next(self.model.parameters()).device)
        return result

    def generate(self, instruction, frames):
        return self.generate_batch([(instruction,frames)])[0]

    def generate_batch(self, requests):
        from .batching import collate_inputs
        if not requests: raise ValueError('Empty generation batch')
        prepared=[self.prepare(instruction,frames) for instruction,frames in requests]
        inputs=(prepared[0]['inputs'] if len(prepared)==1 else
                collate_inputs([item['inputs'] for item in prepared],self.processor.tokenizer.pad_token_id))
        with torch.inference_mode():
            output=self.model.generate(
                **inputs,do_sample=True,temperature=self.temperature,top_p=1.0,top_k=0,
                repetition_penalty=1.0,num_beams=1,max_new_tokens=self.max_response_tokens,
                return_dict_in_generate=True,output_scores=True,output_logits=self.uncertainty,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.pad_token_id)
        results=[]
        eos=self.processor.tokenizer.eos_token_id
        eos=set(eos if isinstance(eos,(list,tuple)) else [eos])
        for index,item in enumerate(prepared):
            ids=output.sequences[index,inputs['input_ids'].shape[-1]:].tolist()
            # HF pads already-finished rows while other rows continue. Include
            # the first EOS in the likelihood and discard only subsequent pads.
            stop=next((i+1 for i,token in enumerate(ids) if token in eos),len(ids))
            ids=ids[:stop]
            if not ids or len(ids)>len(output.scores):
                raise ValueError('Generation scores and response are misaligned')
            probabilities=[float(score[index].float().log_softmax(-1)[token])
                           for score,token in zip(output.scores,ids)]
            actions,uncertainties=None,None
            if self.uncertainty:
                logits=tuple(value[index:index+1] for value in output.logits[:len(ids)])
                actions,uncertainties=extract_action_uncertainties(ids,logits,self.lookup,
                                                                 self.processor.tokenizer.all_special_ids)
            results.append(Generation(
                action_text=self.processor.tokenizer.decode(ids,skip_special_tokens=True),
                prompt_ids=item['inputs']['input_ids'][0].tolist(),response_ids=ids,
                response_logprobs=probabilities,response_mask=[1]*len(ids),
                image_indices=item['image_indices'],uncertainty_actions=actions,uncertainties=uncertainties,
                replay_metadata={'temperature':self.temperature,'raw_prompt':item['prompt'],
                                 'generation_capped':len(ids)==self.max_response_tokens and ids[-1] not in eos,
                                 'observation':asdict(self.observation)}))
        return results
