"""Prepare Python-only Codeforces easy/medium tasks with complete official I/O tests."""
import argparse
import collections
import difflib
import hashlib
import json
from pathlib import Path
import re

from evaluate import read_rows,sha

REPO='open-r1/codeforces'
REVISION='fbe3f6e903ee854eec2e69e9d96d0306cde59baf'
MODEL='Qwen/Qwen2.5-Coder-0.5B-Instruct'
MODEL_REVISION='ea3f2471cf1b1f0db85067f1ef93848e38e88c25'


def normalized(text):return tuple(re.findall(r'\w+|[^\w\s]',text.casefold()))


def same_statement(a,b):
    if a==b:return True
    if min(len(a),len(b))/max(len(a),len(b),1)<0.95:return False
    important=lambda words:[w for w in words if any(c.isdigit() for c in w) or w in {'not','no','non','without','true','false','inclusive','exclusive'}]
    if important(a)!=important(b):return False
    if len(set(a)&set(b))/max(1,len(set(a)|set(b)))<0.95:return False
    return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()>=0.98



def needs_semantic_checker(output_format):
    # A missing metadata checker does not prove a unique answer. Defer explicit
    # alternative-answer/tolerance contracts rather than reward one arbitrary text.
    pattern=r'\b(?:print|output|return)\s+(?:any\b|an arbitrary\b)|\bin\s+any\s+order\b|\bany\s+(?:valid|correct)\s+(?:answer|solution)\b|\b(?:absolute|relative)\s+(?:or\s+(?:absolute|relative)\s+)?error\b'
    return bool(re.search(pattern,output_format or '',re.I))

def exclusion(row,max_rating=1600):
    if not row['executable']:return 'not_upstream_executable'
    if not row['official_tests_complete']:return 'additional_generated_tests_not_loaded'
    if row['input_mode']!='stdio' or row['interaction_format']:return 'file_or_interactive_interface'
    if row['generated_checker']:return 'custom_checker_not_supported_in_initial_scope'
    if row['rating'] is None or not 800<=row['rating']<=max_rating:return 'outside_requested_ratings_or_unrated'
    if 'description' in row and not row.get('description'):return 'missing_problem_statement'
    if 'input_format' in row and (not row.get('input_format') or not row.get('output_format')):return 'missing_input_output_specification'
    if needs_semantic_checker(row.get('output_format')):return 'output_spec_requires_unavailable_semantic_checker'
    return None


def normalize(row):
    cases=row['official_tests']
    if not cases or any(not isinstance(t['input'],str) or not isinstance(t['output'],str) for t in cases):raise ValueError('Missing official I/O cases')
    if len(cases)!=row['testset_size']:raise ValueError('Complete flag disagrees with official test count: '+row['id'])
    text=row['description']+'\n\nInput\n'+row['input_format']+'\n\nOutput\n'+row['output_format']
    for index,example in enumerate(row['examples'] or []):
        text+=f"\n\nExample {index+1}\nInput:\n{example['input']}\nOutput:\n{example['output']}"
    text+=f"\n\nTime limit per input: {row['time_limit']} seconds. Memory limit: {row['memory_limit']} MiB."
    return {'task_id':'codeforces/'+row['id'],'source_id':row['id'],'language':'python','rating':row['rating'],'difficulty':'easy' if row['rating']<=1000 else 'medium','title':row['title'],'instruction':row['description'],
        'prompt':[{'role':'system','content':'Write a complete Python 3 program. Read standard input and write only the required answer to standard output. Return only Python code in one ```python code block. Include required imports. Do not write C++ or another language, explanations, or tests. Any other-language notes in the source do not change the requested language: Python 3.'},{'role':'user','content':text}],
        'tests':{'mode':'stdio','cases':cases,'time_limit':float(row['time_limit']),'memory_limit_mb':int(row['memory_limit'])}}


def assign_split(group):
    value=int(hashlib.sha256(('codeforces-dev-42/'+group).encode()).hexdigest()[:8],16)
    return 'validation' if value%10==0 else 'train'


def prepare(args):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    if args.output.exists():raise ValueError('Output exists; choose a fresh directory')
    evaluation=json.loads((args.evaluation/'manifest.json').read_text())
    if sha(args.evaluation/'protected-statements.jsonl')!=evaluation['protected_statements_sha256']:raise ValueError('Protected statements changed')
    protected=[(r['task_id'],normalized(r['statement'])) for r in read_rows(args.evaluation/'protected-statements.jsonl')]
    candidates=[];decisions=[];files=[];raw_count=0
    fields=['id','aliases','title','rating','description','input_format','output_format','interaction_format','examples','official_tests','official_tests_complete','input_mode','generated_checker','executable','testset_size','time_limit','memory_limit']
    for shard in range(10):
        name=f'verifiable/train-{shard:05d}-of-00010.parquet'
        path=hf_hub_download(REPO,name,repo_type='dataset',revision=REVISION,cache_dir=str(args.cache))
        files.append({'path':name,'sha256':sha(Path(path))})
        file=pq.ParquetFile(path)
        small=file.read(columns=['id','rating','official_tests_complete','input_mode','interaction_format','generated_checker','executable']).to_pylist()
        wanted=set()
        for row in small:
            raw_count+=1;reason=exclusion(row,args.max_rating)
            if reason:decisions.append({'task_id':'codeforces/'+row['id'],'reason':reason})
            else:wanted.add(row['id'])
        for batch in file.iter_batches(batch_size=16,columns=fields,use_threads=False):
            # Convert only wanted rows into Python objects; large unrelated outputs
            # stay in Arrow buffers and are discarded batch by batch.
            for index,identifier in enumerate(batch.column('id').to_pylist()):
                if identifier in wanted:
                    row=batch.slice(index,1).to_pylist()[0]
                    reason=exclusion(row,args.max_rating)
                    if reason:decisions.append({'task_id':'codeforces/'+identifier,'reason':reason})
                    else:candidates.append(row)
        print(f'[prepare] shard {shard+1}/10; candidates={len(candidates)}',flush=True)
    parent={}
    def find(x):
        parent.setdefault(x,x)
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    for row in candidates:
        for alias in row['aliases'] or []:union(row['id'],alias)
        find(row['id'])
    seen=[]
    for row in sorted(candidates,key=lambda r:r['id']):
        key=normalized(row['description'])
        prior=next((other for other,text in seen if same_statement(key,text)),None)
        if prior is not None:union(row['id'],prior)
        else:seen.append((row['id'],key))
    groups=collections.defaultdict(list)
    for row in candidates:groups[find(row['id'])].append(row)
    splits={'train':[],'validation':[]}
    for group,members in sorted(groups.items()):
        overlap=next((identifier for member in members for identifier,text in protected if same_statement(normalized(member['description']),text)),None)
        if overlap is not None:
            decisions.extend({'task_id':'codeforces/'+r['id'],'reason':'protected_statement_overlap','matches':overlap} for r in members);continue
        chosen=sorted(members,key=lambda r:r['id'])[0]
        decisions.extend({'task_id':'codeforces/'+r['id'],'reason':'duplicate_or_published_alias','matches':'codeforces/'+chosen['id']} for r in members if r['id']!=chosen['id'])
        item=normalize(chosen);item['group_id']=group
        splits[assign_split(group)].append(item)
    if not splits['train'] or not splits['validation']:raise ValueError('Empty training/development split')
    assert raw_count==sum(map(len,splits.values()))+len(decisions)
    args.output.mkdir(parents=True)
    for split,rows in splits.items():(args.output/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (args.output/'exclusions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decisions))
    manifest={'protocol':'python-grpo-v1','source':{'repo':REPO,'revision':REVISION,'split':'verifiable/train','files':files},'model':MODEL,'revision':MODEL_REVISION,'language':'python','execution_mode':'stdio','execution_image':'python-grpo-cpython:v3','container_memory':'2g','max_completion_tokens':1024,
        'raw_rows':raw_count,'candidate_rows':len(candidates),'requested_rating_range':[800,args.max_rating],'difficulty_definition':{'easy':[800,1000],'medium':[1100,args.max_rating]},'counts':{s:len(r) for s,r in splits.items()},'difficulty_counts':{s:dict(collections.Counter(r['difficulty'] for r in rows)) for s,rows in splits.items()},'official_test_records':{s:sum(len(r['tests']['cases']) for r in rows) for s,rows in splits.items()},'exclusions':dict(collections.Counter(r['reason'] for r in decisions)),
        'split_rule':'Union published aliases and near-identical statements before splitting; SHA-256 group ID with seed 42, modulo 10 gives a fixed roughly 10% development set; no model outcomes used.',
        'files_sha256':{s:sha(args.output/f'{s}.jsonl') for s in splits},'evaluation_manifest':str(args.evaluation/'manifest.json'),'evaluation_manifest_sha256':sha(args.evaluation/'manifest.json'),'confirmation':evaluation['confirmation'],'codeforces_test_split_read':False,
        'training_reward':'Binary: all official I/O records pass; fail fast on first failure. No minimum test count or model-difficulty calibration gate.','development_protocol':'Fixed Codeforces train-derived development split, same official Python I/O grader before/after. Existing MBPP development/confirmation inputs remain unchanged.',
        'limitations':['Easy/medium rating labels are an experiment choice, not a guarantee of suitability for the 0.5B model.','Upstream complete official tests are not necessarily exhaustive; missing checker metadata does not prove unique outputs. Explicit alternate-answer/tolerance specifications are deferred.','Default no-checker comparison follows upstream whitespace/YES-NO normalization, not general semantic equivalence.','Per-case source time/memory limits are applied to Python; this is not an exact reproduction of Codeforces language-specific judging.','Text and alias checks do not guarantee semantic or pretraining decontamination.']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['candidate_rows','counts','difficulty_counts','official_test_records','exclusions']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('data/codeforces-python'));p.add_argument('--evaluation',type=Path,default=Path('evaluation'));p.add_argument('--max-rating',type=int,default=1600);p.add_argument('--cache',type=Path,default=Path('.cache/huggingface/hub').resolve())
    prepare(p.parse_args())
