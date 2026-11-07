"""Download a deterministic revision-pinned OpenCoder educational sample."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))


def main(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Import output already exists')
    args.output.mkdir(parents=True,exist_ok=True)
    repo='OpenCoder-LLM/opc-sft-stage2';subset='educational_instruct'
    revision=args.revision or HfApi().dataset_info(repo).sha
    source=load_dataset(repo,subset,split='train',revision=revision,streaming=True)
    rows=list(source.shuffle(seed=args.seed,buffer_size=args.buffer).take(args.rows))
    raw=args.output/'raw.jsonl';raw.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest={'source':repo,'subset':subset,'revision':revision,'license_as_declared':'mit','rows':len(rows),'sampling':{'seed':args.seed,'shuffle_buffer':args.buffer,'take':args.rows},'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'source_seq_ids':[row['seq_id'] for row in rows],'created_at':time.time()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:v for k,v in manifest.items() if k!='source_seq_ids'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/opencoder-candidates'));p.add_argument('--revision');p.add_argument('--rows',type=int,default=3000);p.add_argument('--seed',type=int,default=42);p.add_argument('--buffer',type=int,default=10000)
    main(p.parse_args())
