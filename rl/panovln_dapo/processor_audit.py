"""Replay real collected observations through original/exported processors on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoProcessor
from .actor import load_batch,replay


def audit(initial,candidate,batch):
    progress=json.loads((Path(candidate)/'vln_progress.json').read_text())
    if hashlib.sha256(Path(batch).read_bytes()).hexdigest()!=progress['source_batch_sha256']:
        raise ValueError('Processor audit batch differs from checkpoint source')
    _,rows=load_batch(batch)
    original=AutoProcessor.from_pretrained(initial,local_files_only=True)
    exported=AutoProcessor.from_pretrained(candidate,local_files_only=True)
    selected=sorted({0,max(range(len(rows)),key=lambda index:rows[index]['frame_count'])})
    checks=[]
    for index in selected:
        before=replay(original,rows[index],'cpu')
        after=replay(exported,rows[index],'cpu')
        if before.keys()!=after.keys():raise ValueError('Processor output fields changed')
        for key in before:
            left,right=before[key],after[key]
            if torch.is_tensor(left):
                if not torch.is_tensor(right) or left.dtype!=right.dtype or not torch.equal(left,right):
                    raise ValueError(f'Processor tensor changed: row={index}, field={key}')
            elif left!=right:raise ValueError(f'Processor metadata changed: {key}')
        checks.append({'row':index,'history_frames':rows[index]['frame_count'],
                       'selected_images':rows[index]['generation'].image_indices,
                       'prompt_tokens':len(rows[index]['generation'].prompt_ids),'fields':sorted(before)})
    return {'scope':'CPU processor roundtrip on real first/long-history replay observations',
            'exact_tensor_equality':True,'checks':checks}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--initial',required=True)
    p.add_argument('--candidate',required=True)
    p.add_argument('--batch',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(2)
    report=audit(args.initial,args.candidate,args.batch)
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
