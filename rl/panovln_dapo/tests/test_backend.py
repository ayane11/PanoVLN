import unittest
from types import SimpleNamespace
import torch
from panovln_dapo.backend import response_logprobs


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transition = torch.nn.Parameter(torch.arange(25).reshape(5,5).float()/7)
        self.seen = None

    def forward(self, **kwargs):
        self.seen = kwargs
        logits = self.transition[kwargs['input_ids']]
        return SimpleNamespace(logits=logits[:, -kwargs['logits_to_keep']:])


class LikelihoodTests(unittest.TestCase):
    def test_response_shift_temperature_and_grad(self):
        model = TinyCausalModel()
        prompt = {'input_ids':torch.tensor([[0,1]]), 'attention_mask':torch.ones(1,2,dtype=torch.long),
                  'mm_token_type_ids':torch.tensor([[1,0]]), 'image_current_index':torch.tensor([0])}
        actual = response_logprobs(model,prompt,[2,3],temperature=1.5)
        expected = torch.stack([torch.log_softmax(model.transition[1]/1.5,0)[2],
                                torch.log_softmax(model.transition[2]/1.5,0)[3]])
        torch.testing.assert_close(actual,expected)
        actual.sum().backward()
        self.assertGreater(model.transition.grad[1:3].abs().sum().item(),0)
        self.assertEqual(model.transition.grad[3].abs().sum().item(),0)
        self.assertEqual(model.seen['mm_token_type_ids'].tolist(),[[1,0,0,0]])
        self.assertEqual(model.seen['attention_mask'].tolist(),[[1,1,1,1]])
        self.assertIs(model.seen['image_current_index'],prompt['image_current_index'])
        self.assertEqual(prompt['input_ids'].shape[-1],2)

    def test_empty_response_rejected(self):
        with self.assertRaisesRegex(ValueError,'empty'):
            response_logprobs(TinyCausalModel(),{'input_ids':torch.tensor([[0]]),'attention_mask':torch.ones(1,1)},[])

    def test_entropy_matches_categorical_value_gradient_and_mask(self):
        model=TinyCausalModel()
        prompt={'input_ids':torch.tensor([[0,1]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
        logp,entropy=response_logprobs(model,prompt,[2,3],temperature=.7,return_entropy=True)
        reference=model.transition.detach().clone().requires_grad_()
        distribution=torch.distributions.Categorical(logits=reference[[1,2]]/.7)
        expected=distribution.entropy()
        torch.testing.assert_close(entropy,expected)
        torch.testing.assert_close(logp,distribution.log_prob(torch.tensor([2,3])))
        mask=torch.tensor([1.,0.])
        actual_loss=((-logp-.03*entropy)*mask).sum()/mask.sum()
        expected_loss=((-distribution.log_prob(torch.tensor([2,3]))-.03*expected)*mask).sum()/mask.sum()
        actual_loss.backward();expected_loss.backward()
        torch.testing.assert_close(model.transition.grad,reference.grad)
        self.assertEqual(model.transition.grad[2].abs().sum().item(),0.)

    def test_entropy_matches_current_source_formula(self):
        import ast
        from pathlib import Path
        path=Path(__file__).resolve().parents[4]/'SimpleVLN/RL/verl/utils/torch_functional.py'
        source=ast.parse(path.read_text())
        definition=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='entropy_from_logits')
        namespace={'torch':torch}
        exec(compile(ast.Module(body=[definition],type_ignores=[]),str(path),'exec'),namespace)
        model=TinyCausalModel()
        prompt={'input_ids':torch.tensor([[0,1]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
        _,entropy=response_logprobs(model,prompt,[2,3],temperature=1.5,return_entropy=True)
        source_entropy=namespace['entropy_from_logits'](model.transition[[1,2]]/1.5)
        torch.testing.assert_close(entropy,source_entropy,rtol=0,atol=0)
        grad=torch.autograd.grad(entropy.sum(),model.transition,retain_graph=True)[0]
        source_grad=torch.autograd.grad(source_entropy.sum(),model.transition)[0]
        torch.testing.assert_close(grad,source_grad,rtol=1e-5,atol=2e-8)

class BatchedLikelihoodTests(unittest.TestCase):
    def test_variable_lengths_temperatures_entropy_and_gradient(self):
        from panovln_dapo.backend import batch_response_logprobs
        batched=TinyCausalModel();reference=TinyCausalModel()
        prompts=[{'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long),
                  'mm_token_type_ids':torch.zeros(1,len(ids),dtype=torch.long)} for ids in ([0,1],[2],[3,1,2])]
        responses=[[2,3],[1],[4,0,1]];temperatures=[.7,1.,1.5]
        actual=batch_response_logprobs(batched,prompts,responses,temperatures,0,return_entropy=True)
        expected=[response_logprobs(reference,p,r,t,return_entropy=True) for p,r,t in zip(prompts,responses,temperatures)]
        for pair,ref in zip(actual,expected):
            for value,other in zip(pair,ref):torch.testing.assert_close(value,other)
        masks=[torch.tensor([1.,0.]),torch.ones(1),torch.tensor([1.,1.,0.])]
        def loss(values):
            return sum(((-logp-.03*entropy)*mask).sum() for (logp,entropy),mask in zip(values,masks))/4
        loss(actual).backward();loss(expected).backward()
        torch.testing.assert_close(batched.transition.grad,reference.transition.grad)
        self.assertEqual(batched.seen['attention_mask'].sum(1).tolist(),[4,2,6])

    def test_geometry_and_local_image_indices(self):
        from panovln_dapo.backend import batch_response_logprobs
        model=TinyCausalModel();prompts=[]
        for count in (1,3):
            prompts.append({'input_ids':torch.tensor([[1]]),'attention_mask':torch.ones(1,1,dtype=torch.long),
                            'image_num_images':torch.tensor([count]),'image_current_index':torch.tensor([count-1]),
                            'image_erp_geometry':torch.arange(count*2).reshape(count,2).float()})
        batch_response_logprobs(model,prompts,[[2],[3]],[1.,1.],0)
        self.assertEqual(model.seen['image_current_index'].tolist(),[0,2])
        self.assertEqual(model.seen['image_num_images'].tolist(),[1,3])
        self.assertEqual(model.seen['image_erp_geometry'].dtype,torch.float32)

class BatchedGenerationTests(unittest.TestCase):
    def test_eos_padding_scores_and_original_prompts(self):
        from panovln_dapo.backend import NativePolicy
        class Tokenizer:
            eos_token_id=4
            pad_token_id=0
            def decode(self,ids,**kw):return str(ids)
        class Model:
            def generate(self,**kw):
                self.seen=kw
                return SimpleNamespace(sequences=torch.tensor([[0,1,2,4,0],[2,3,1,2,4]]),
                                       scores=tuple(torch.arange(10).reshape(2,5).float()+i for i in range(3)))
        model=Model();policy=NativePolicy(model,SimpleNamespace(tokenizer=Tokenizer()),max_response_tokens=3)
        def prepare(instruction,frames):
            ids=[1] if instruction=='a' else [2,3]
            return dict(inputs={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones(1,len(ids),dtype=torch.long)},
                        image_indices=[len(frames)],prompt=instruction)
        policy.prepare=prepare
        first,second=policy.generate_batch([('a',[]),('b',[0])])
        self.assertEqual(first.response_ids,[2,4]);self.assertEqual(second.response_ids,[1,2,4])
        self.assertEqual(first.prompt_ids,[1]);self.assertEqual(second.prompt_ids,[2,3])
        self.assertEqual(first.image_indices,[0]);self.assertEqual(second.image_indices,[1])
        self.assertEqual(first.response_mask,[1,1])
        self.assertFalse(first.replay_metadata['generation_capped'])
        self.assertFalse(second.replay_metadata['generation_capped'])
        torch.testing.assert_close(torch.tensor(first.response_logprobs),torch.log_softmax(torch.arange(5).float(),0)[torch.tensor([2,4])])
        self.assertFalse(model.seen['output_logits'])

class EntropyCompileTests(unittest.TestCase):
    def test_compile_only_used_when_requested_and_entropy_enabled(self):
        from unittest.mock import patch
        from panovln_dapo.backend import entropy_from_logits
        model=TinyCausalModel();model._dapo_compile_entropy=True
        prompt={'input_ids':torch.tensor([[0,1]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
        with patch('panovln_dapo.backend.compiled_entropy',return_value=entropy_from_logits) as compile_fn:
            response_logprobs(model,prompt,[2])
            compile_fn.assert_not_called()
            _,entropy=response_logprobs(model,prompt,[2],return_entropy=True)
            compile_fn.assert_called_once()
            torch.testing.assert_close(entropy,entropy_from_logits(model.transition[1:2]))
            model._dapo_compile_entropy=False
            response_logprobs(model,prompt,[2],return_entropy=True)
            self.assertEqual(compile_fn.call_count,1)
