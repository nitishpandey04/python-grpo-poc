"""Normalization, conservative overlap checks and transparent quality tiers."""
import ast
import builtins
import collections
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import threading
import time

from educational_oracles import independent_cases,matching_oracle
from execution import Sandbox
from task_selection import read_rows,sha

STOP={'write','a','an','the','python','function','given','that','to','of','and','please','implement','takes','input','as','in','is','for','it','should','which'}


def prompt_tokens(text):
    return set(re.findall(r'[a-z0-9]+',text.lower()))-STOP


def similarity(a,b):
    if not a or not b:return 0.0
    if min(len(a),len(b))/max(len(a),len(b))<0.75:return 0.0
    overlap=len(a&b);return overlap/(len(a)+len(b)-overlap)


class CanonicalNames(ast.NodeTransformer):
    def __init__(self):self.names={}
    def name(self,name):
        if name not in self.names:self.names[name]='v'+str(len(self.names))
        return self.names[name]
    def visit_FunctionDef(self,node):
        node.name=self.name(node.name);node.returns=None;node.type_comment=None
        node.body=[n for n in node.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
        return self.generic_visit(node)
    def visit_arg(self,node):node.arg=self.name(node.arg);node.annotation=None;return node
    def visit_Name(self,node):
        if node.id in self.names or node.id not in vars(builtins):node.id=self.name(node.id)
        return node


def code_key(code):
    tree=ast.parse(code)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    return hashlib.sha256(ast.dump(CanonicalNames().visit(tree),include_attributes=False).encode()).hexdigest()


def literal_examples(tests,entry):
    examples=[]
    for source in tests:
        tree=ast.parse(source)
        if len(tree.body)!=1 or not isinstance(tree.body[0],ast.Assert):raise ValueError('tests_require_setup_or_unsupported_contract')
        expr=tree.body[0].test
        if not isinstance(expr,ast.Compare) or len(expr.ops)!=1 or not isinstance(expr.ops[0],(ast.Eq,ast.Is)):
            raise ValueError('unsupported_test_contract')
        left,right=expr.left,expr.comparators[0]
        if isinstance(right,ast.Call) and isinstance(right.func,ast.Name) and right.func.id==entry:left,right=right,left
        if not isinstance(left,ast.Call) or not isinstance(left.func,ast.Name) or left.func.id!=entry:raise ValueError('unsupported_test_contract')
        args=tuple(ast.literal_eval(x) for x in left.args)
        kwargs={kw.arg:ast.literal_eval(kw.value) for kw in left.keywords}
        if None in kwargs:raise ValueError('unsupported_test_contract')
        expected=ast.literal_eval(right)
        if len(repr((args,kwargs,expected)))>4000:raise ValueError('large_test_fixture')
        examples.append((args,kwargs,expected))
    return examples


def normalize(raw):
    code=raw['code'].strip();entry=raw['entry_point'].strip();text=raw['instruction'].strip()
    if not text or len(text)>2500 or len(code)>2200:raise ValueError('outside_short_function_scope')
    tree=ast.parse(code)
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    if entry not in functions or not entry.isidentifier():raise ValueError('missing_entry_point')
    if any(isinstance(n,(ast.ClassDef,ast.AsyncFunctionDef)) for n in ast.walk(tree)):raise ValueError('outside_short_function_scope')
    for node in tree.body:
        if not isinstance(node,(ast.FunctionDef,ast.Import,ast.ImportFrom,ast.Assign,ast.AnnAssign)):
            if not (isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):raise ValueError('top_level_execution')
    tests=list(dict.fromkeys(t.strip() for t in raw['testcase'] if t.strip()))
    if not 3<=len(tests)<=40:raise ValueError('test_count_outside_scope')
    examples=literal_examples(tests,entry)
    if len({repr((args,kwargs)) for args,kwargs,_ in examples})<3:raise ValueError('insufficient_distinct_inputs')
    function=functions[entry]
    signature=f'def {entry}({ast.unparse(function.args)}):'
    return {'task_id':'opencoder/'+str(raw['seq_id']),'source_seq_id':raw['seq_id'],'instruction':text,'reference':code,'entry_point':entry,'signature':signature,'provided_tests':tests,'examples':examples,'code_key':code_key(code),'tokens':prompt_tokens(text)}


def assertion_suite_score(sandbox,completion,tests,task_id=None):
    score=sandbox.score(completion,['\n'.join(tests)])
    return {**score,'correct':score['passed']==score['total'],'test_assertions':len(tests),'correctness_protocol':'bundled_assertions'}


def contract_group(family):
    if not family:return None
    if family.startswith('palindrome_'):return 'palindrome'
    if family in {'is_prime','not_prime'}:return 'primality'
    if family in {'sum_even_values','sum_odd_values'}:return 'parity_sum'
    return family


def split_group(key):
    bucket=int(hashlib.sha256(('opencoder-split-42/'+key).encode()).hexdigest()[:8],16)%100
    return 'train' if bucket<80 else 'validation' if bucket<90 else 'confirmation'


def curate(args):
    if args.output.exists():raise ValueError('Curation output exists; choose a new directory')
    imported=json.loads((args.source/'manifest.json').read_text())
    if sha(args.source/'raw.jsonl')!=imported['raw_sha256']:raise ValueError('Raw import changed')
    raw=[json.loads(l) for l in (args.source/'raw.jsonl').read_text().splitlines()]
    args.output.mkdir(parents=True)
    protected=[];protected_families=set()
    for path in [args.protected/'validation.jsonl',args.protected/'test.jsonl']:
        for row in read_rows(path):
            text=row['prompt'][-1]['content'].split('\n\nRequired signature:')[0]
            protected.append((prompt_tokens(text),code_key(row['reference'])))
            try:
                entry=ast.parse(row['signature']+'\n pass').body[0].name
                match=matching_oracle(text,literal_examples(row['tests'],entry))
                if match:protected_families.add(contract_group(match[0]))
            except (ValueError,TypeError,SyntaxError,KeyError):pass
    protected_keys={key for _,key in protected}
    groups=[];rejections=[]
    for item in raw:
        try:row=normalize(item)
        except (ValueError,SyntaxError,TypeError,RecursionError,KeyError) as exc:
            rejections.append({'source_seq_id':item.get('seq_id'),'reason':str(exc)});continue
        match=matching_oracle(row['instruction'],row['examples'])
        row['contract_group']=contract_group(match[0]) if match else None
        if row['contract_group'] in protected_families or row['code_key'] in protected_keys or any(similarity(row['tokens'],tokens)>=0.8 for tokens,_ in protected):
            rejections.append({'source_seq_id':row['source_seq_id'],'reason':'heldout_code_or_near_prompt_overlap'});continue
        matches=[group for group in groups if row['code_key'] in group['code_keys'] or (row['contract_group'] is not None and row['contract_group'] in group['families']) or any(similarity(row['tokens'],other['tokens'])>=0.8 for other in group['members'])]
        if matches:
            group=matches[0];group['members'].append(row);group['code_keys'].add(row['code_key']);group['families'].add(row['contract_group'])
            for other in matches[1:]:
                group['members'].extend(other['members']);group['code_keys'].update(other['code_keys']);group['families'].update(other['families']);groups.remove(other)
        else:groups.append({'members':[row],'code_keys':{row['code_key']},'families':{row['contract_group']}})
    print(f'Normalized into {len(groups)} conservative problem groups after {len(rejections)} row exclusions',flush=True)
    connections=[];lock=threading.Lock();local=threading.local()
    def init():
        local.sandbox=Sandbox();local.sandbox.start()
        with lock:connections.append(local.sandbox)
    def check_group(group):
        rejected=[]
        # The representative is selected by reference/test quality, never model outcomes.
        group_key=hashlib.sha256('|'.join(sorted(group['code_keys'])).encode()).hexdigest()
        for row in sorted(group['members'],key=lambda r:(-len(r['provided_tests']),len(r['reference']),r['source_seq_id'])):
            base=assertion_suite_score(local.sandbox,row['reference'],row['provided_tests'])
            if not base['correct']:
                rejected.append({'source_seq_id':row['source_seq_id'],'reason':'reference_failed_supplied_tests','status':base['status']});continue
            family,extra=independent_cases(row['instruction'],row['examples'],seed=row['source_seq_id']%2147483647)
            tests=list(row['provided_tests'])
            if extra:
                tests += [f"assert {row['entry_point']}({', '.join(repr(x) for x in inputs)}) == {expected!r}" for inputs,expected in extra]
                tests=list(dict.fromkeys(tests))
                extended=assertion_suite_score(local.sandbox,row['reference'],tests)
                if not extended['correct']:
                    rejected.append({'source_seq_id':row['source_seq_id'],'reason':'independent_contract_reference_disagreement','family':family,'status':extended['status']});continue
            elif len(tests)<4:
                rejected.append({'source_seq_id':row['source_seq_id'],'reason':'too_few_tests_without_independent_augmentation'});continue
            controls=['None','False','True','0','1','-1',"''",'[]','{}']
            survivors=[]
            for value in controls:
                mutant=f"def {row['entry_point']}(*args, **kwargs):\n return {value}"
                if assertion_suite_score(local.sandbox,mutant,tests)['correct']:survivors.append(value)
            if survivors:
                rejected.append({'source_seq_id':row['source_seq_id'],'reason':'tests_accept_constant_control','survivors':survivors});continue
            tier='independent_contract_checked' if extra else 'reference_and_constant_control_screened'
            public={k:v for k,v in row.items() if k not in {'tokens','examples','code_key','provided_tests'}}
            public.update(tests=tests,group_id=group_key,split=split_group(group_key),quality_tier=tier,independent_family=family,provided_assertions=len(row['provided_tests']),additional_assertions=len(tests)-len(row['provided_tests']),constant_controls_rejected=len(controls),reward_protocol='assert_suite',source_revision=imported['revision'],source_license='mit',needs_semantic_review=not bool(extra))
            public['prompt']=[{'role':'system','content':'Write correct Python functions. Return only Python code in a single ```python code block. Include the requested function and any helper functions. Do not include example calls, print statements, or explanations.'},{'role':'user','content':row['instruction']+'\n\nRequired signature:\n'+row['signature']}]
            return {'accepted':public,'rejected':rejected,'group_rows':len(group['members'])}
        return {'accepted':None,'rejected':rejected,'group_rows':len(group['members'])}
    accepted=[];group_records=[]
    try:
        with ThreadPoolExecutor(max_workers=args.workers,initializer=init) as pool:
            for i,result in enumerate(pool.map(check_group,groups)):
                group_records.append(result)
                if result['accepted']:accepted.append(result['accepted'])
                with (args.output/'group-checks.jsonl').open('a') as handle:handle.write(json.dumps(result)+'\n')
                (args.output/'status.json').write_text(json.dumps({'phase':'curating','groups_completed':i+1,'groups_total':len(groups),'accepted':len(accepted),'updated_at':time.time()}))
                print(f"[curation] {i+1}/{len(groups)} groups; accepted={len(accepted)}",flush=True)
    finally:
        for sandbox in connections:sandbox.close()
    for split in ['train','validation','confirmation']:
        (args.output/f'{split}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in accepted if row['split']==split))
    (args.output/'row-exclusions.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rejections))
    exclusions=collections.Counter(row['reason'] for row in rejections)
    exclusions.update(row['reason'] for result in group_records for row in result['rejected'])
    image=__import__('subprocess').run(['docker','image','inspect','--format','{{.Id}}','python-grpo-cpython:v2'],capture_output=True,text=True,check=True).stdout.strip()
    manifest={'protocol':'opencoder-screened-v1','source':imported,'image_id':image,'protected_source_sha256':{split:sha(args.protected/f'{split}.jsonl') for split in ['validation','test']},'raw_candidates':len(raw),'problem_groups':len(groups),'retained_groups':len(accepted),'duplicate_or_related_rows_collapsed':sum(len(g['members'])-1 for g in groups),'counts_after':{split:sum(r['split']==split for r in accepted) for split in ['train','validation','confirmation']},'quality_tiers':dict(collections.Counter(r['quality_tier'] for r in accepted)),'exclusions':dict(exclusions),'files_sha256':{split:sha(args.output/f'{split}.jsonl') for split in ['train','validation','confirmation']},'protected_contract_groups':sorted(protected_families),'split_rule':'SHA-256 group hash, seed 42: 80/10/10; AST clones, near prompts and recognized contract families grouped before model calibration','caveats':['Supplied reference/test agreement and constant-control rejection are screening, not independent semantic verification.','Independent-contract tiers apply only to explicit narrow oracle contracts and generated cases.','Conservative AST/prompt similarity does not guarantee absence of semantic or pretraining overlap.','Do not train on screened-only tasks without semantic review or further test strengthening.'],'confirmation_predictions_generated':False,'created_at':time.time()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'status.json').write_text(json.dumps({'phase':'complete','counts':manifest['counts_after'],'updated_at':time.time()}))
    print(json.dumps({k:v for k,v in manifest.items() if k!='source'},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('data/opencoder-candidates'));p.add_argument('--output',type=Path,default=Path('runs/opencoder-curation-v1'));p.add_argument('--protected',type=Path,default=Path('data/prepared'));p.add_argument('--workers',type=int,default=4)
    curate(p.parse_args())
