"""Check exported model tensors against the policy used to collect the batch.

This is an on-disk integrity/update check; it does not substitute for model
forward, optimizer resume, or navigation evaluation.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open


def weight_map(directory):
    directory=Path(directory)
    index=directory/'model.safetensors.index.json'
    if index.exists():
        mapping=json.loads(index.read_text())['weight_map']
    else:
        with safe_open(directory/'model.safetensors',framework='pt',device='cpu') as handle:
            mapping={key:'model.safetensors' for key in handle.keys()}
    if not mapping:raise ValueError('Empty model checkpoint')
    for shard in set(mapping.values()):
        path=(directory/shard).resolve()
        if path.parent!=directory.resolve() or not path.is_file():
            raise ValueError(f'Invalid model shard: {shard}')
    return mapping


def canonical_keys(mapping):
    """Normalize observed legacy Pano/HF names, rejecting ambiguous aliases."""
    result={}
    for source,shard in mapping.items():
        key=source
        while key.startswith('model.language_model.language_model.'):
            key=key.replace('model.language_model.language_model.','model.language_model.',1)
        if key.startswith('model.language_model.visual.'):
            key=key.replace('model.language_model.visual.','model.visual.',1)
        if key in result:raise ValueError(f'Ambiguous checkpoint key normalization: {key}')
        result[key]=(shard,source)
    return result


def validate_tied_head(directory, mapping):
    config_path=directory/'config.json'
    if not config_path.exists():return False
    config=json.loads(config_path.read_text())
    if config.get('tie_word_embeddings') is not True:return False
    embedding='model.language_model.embed_tokens.weight'
    if embedding not in mapping:raise ValueError('Tied checkpoint lacks embedding tensor')
    if 'lm_head.weight' in mapping:
        head_shard,head_key=mapping['lm_head.weight']
        embed_shard,embed_key=mapping[embedding]
        with safe_open(directory/head_shard,framework='pt',device='cpu') as h, safe_open(directory/embed_shard,framework='pt',device='cpu') as e:
            if not torch.equal(h.get_tensor(head_key),e.get_tensor(embed_key)):
                raise ValueError('Tied head and embedding tensors disagree')
    return True


def compare_tensor_chunks(old, new, key, *, chunk_elements=1_048_576):
    """Full coverage with bounded scratch space; never sample tensor contents."""
    if old.shape != new.shape:
        raise ValueError(f'Tensor shape changed: {key}')
    if chunk_elements < 1:
        raise ValueError('chunk_elements must be positive')
    old_flat, new_flat = old.reshape(-1), new.reshape(-1)
    changed = cast_changed = False
    for start in range(0, new_flat.numel(), chunk_elements):
        left = old_flat[start:start+chunk_elements]
        right = new_flat[start:start+chunk_elements]
        if not torch.isfinite(right).all():
            raise ValueError(f'Nonfinite exported tensor: {key}')
        different = not torch.equal(left, right)
        changed |= different
        if different and not cast_changed:
            cast_changed = not torch.equal(left, right.to(dtype=left.dtype))
    return changed, cast_changed


def audit(initial, candidate, *, expected_updates=1):
    if not isinstance(expected_updates,int) or expected_updates<1:raise ValueError('Invalid checkpoint update interval')
    initial,candidate=Path(initial),Path(candidate)
    old_map,new_map=canonical_keys(weight_map(initial)),canonical_keys(weight_map(candidate))
    old_tied,new_tied=validate_tied_head(initial,old_map),validate_tied_head(candidate,new_map)
    if old_tied!=new_tied:raise ValueError('Tied embedding configuration changed')
    if old_tied:
        old_map.pop('lm_head.weight',None);new_map.pop('lm_head.weight',None)
    if old_map.keys()!=new_map.keys():
        raise ValueError(f'Checkpoint tensor keys differ: missing={sorted(old_map.keys()-new_map.keys())[:8]}, extra={sorted(new_map.keys()-old_map.keys())[:8]}')
    progress=json.loads((candidate/'vln_progress.json').read_text())
    data=(candidate/'data.pt').read_bytes()
    if hashlib.sha256(data).hexdigest()!=progress['data_sha256']:
        raise ValueError('Data/RNG checkpoint checksum mismatch')
    if not (candidate/'optimizer.pt').is_file() or (candidate/'optimizer.pt').stat().st_size==0:
        raise ValueError('Missing optimizer checkpoint')
    previous=initial/'vln_progress.json'
    prior=json.loads(previous.read_text())['policy_version'] if previous.exists() else 0
    expected=prior+expected_updates
    if progress['policy_version']!=expected:raise ValueError('Policy version did not advance by the expected update interval')
    if expected_updates>1 and progress.get('previous_durable_policy_version')!=prior:
        raise ValueError('Periodic checkpoint is not bound to previous durable policy')
    if progress['optimizer_steps']<1 or progress['changed_parameter_samples']<1:
        raise ValueError('No reported optimizer update')
    # Open one pair of shards at a time and read each tensor, limiting peak RAM.
    pairs={}
    for key in old_map:pairs.setdefault((old_map[key][0],new_map[key][0]),[]).append(key)
    counts=Counter();changed_examples=[]
    for (old_shard,new_shard),keys in pairs.items():
        with safe_open(initial/old_shard,framework='pt',device='cpu') as left, safe_open(candidate/new_shard,framework='pt',device='cpu') as right:
            for key in keys:
                old,new=left.get_tensor(old_map[key][1]),right.get_tensor(new_map[key][1])
                changed,cast_changed=compare_tensor_chunks(old,new,key)
                frozen=key.startswith('panovggt.')
                counts['tensors']+=1
                counts['geometry_tensors' if frozen else 'policy_tensors']+=1
                if changed:
                    counts['changed_tensors']+=1
                    if frozen:raise ValueError(f'Frozen geometry changed: {key}')
                    if cast_changed:
                        counts['changed_tensors_after_original_dtype_cast']+=1
                    if len(changed_examples)<12:changed_examples.append(key)
                if new.dtype!=old.dtype:counts['dtype_changes']+=1
    if counts['geometry_tensors']==0:raise ValueError('Export lacks embedded geometry encoder')
    if counts['changed_tensors']==0:raise ValueError('Exported policy tensors did not change')
    return {'scope':'On-disk tensor integrity; forward/resume/evaluation still required',
            'initial':str(initial),'candidate':str(candidate),'counts':dict(counts),
            'changed_examples':changed_examples,'policy_version':progress['policy_version'],
            'optimizer_steps':progress['optimizer_steps'],'frozen_geometry_unchanged':True,
            'verified_tied_head_alias':old_tied,'key_normalization':'Observed legacy language_model prefixes only; collisions rejected'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',required=True,type=Path)
    parser.add_argument('--candidate',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();torch.set_num_threads(2)
    report=audit(args.initial,args.candidate)
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
