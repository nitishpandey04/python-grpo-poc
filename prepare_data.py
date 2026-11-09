"""Prepare native OpenCoder test programs without reference or difficulty gates."""
import argparse
import ast
import collections
import difflib
import json
from pathlib import Path
import re

from evaluate import read_rows,sha

MODEL='Qwen/Qwen2.5-Coder-0.5B-Instruct'
REVISION='ea3f2471cf1b1f0db85067f1ef93848e38e88c25'
# These supplied programs use root before initialization. Preserve the original
# rows in the raw download and record their IDs instead of rewriting their tests.
BROKEN_TESTS={37809597094,23726331439,24060963630}


def normalized(text):
    return tuple(re.findall(r'\w+|[^\w\s]',text.casefold()))


def same_statement(a,b):
    if a==b:return True
    if min(len(a),len(b))/max(len(a),len(b),1)<0.95:return False
    # Do not merge numerically different or negated contracts as punctuation edits.
    important=lambda words:[w for w in words if any(c.isdigit() for c in w) or w in {'not','no','non','without','true','false','inclusive','exclusive'}]
    if important(a)!=important(b):return False
    if len(set(a)&set(b))/max(1,len(set(a)|set(b)))<0.95:return False
    return difflib.SequenceMatcher(None,a,b,autojunk=False).ratio()>=0.98


def normalize(raw):
    statement=raw.get('instruction','').strip()
    if not statement:raise ValueError('missing_statement')
    tests=raw.get('testcase')
    if not isinstance(tests,list) or not tests or not all(isinstance(t,str) for t in tests):raise ValueError('missing_test_program')
    program='\n'.join(tests)
    if not program.strip():raise ValueError('missing_test_program')
    try:
        tree=ast.parse(program)
        compile(tree,'<dataset-tests>','exec')
    except SyntaxError as exc:raise ValueError('test_syntax_error') from exc
    # A program with only setup/printing supplies no pass/fail check. There is
    # no minimum number or required shape of checks; setup and helpers are kept.
    if not any(isinstance(n,(ast.Assert,ast.Raise)) for n in ast.walk(tree)):
        raise ValueError('no_explicit_test_check')
    entry=raw.get('entry_point','').strip()
    interface=''
    reference=raw.get('code') or ''
    # Signature extraction is optional metadata, never an eligibility test.
    try:
        for node in ast.parse(reference).body:
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name==entry:
                interface=('async ' if isinstance(node,ast.AsyncFunctionDef) else '')+f'def {entry}({ast.unparse(node.args)}):';break
    except SyntaxError:pass
    if not interface and entry:interface='Requested entry point: '+entry
    prompt=statement+('\n\n'+interface if interface else '')
    return {'task_id':'opencoder/'+str(raw['seq_id']),'instruction':statement,'entry_point':entry,
            'prompt':[{'role':'system','content':'Write correct Python code. Return only Python code in a single ```python code block, including any required helpers and imports.'},{'role':'user','content':prompt}],
            'tests':[program],'reference':reference}


def prepare(args):
    if args.output.exists():raise ValueError('Output exists; choose a fresh directory')
    imported=json.loads(args.raw.with_name('manifest.json').read_text())
    if sha(args.raw)!=imported['raw_sha256']:raise ValueError('Raw download changed')
    evaluation=json.loads((args.evaluation/'manifest.json').read_text())
    if sha(args.evaluation/'development.jsonl')!=evaluation['development_sha256']:raise ValueError('Fixed development data changed')
    if sha(args.evaluation/'protected-statements.jsonl')!=evaluation['protected_statements_sha256']:raise ValueError('Protected statement data changed')
    protected=[(r['task_id'],normalized(r['statement'])) for r in read_rows(args.evaluation/'protected-statements.jsonl')]
    reserved={identifier for identifier,_ in protected if isinstance(identifier,str) and identifier.startswith('opencoder/')}
    accepted=[];decisions=[];seen=[]
    raw=[json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    for i,item in enumerate(raw):
        task='opencoder/'+str(item['seq_id'])
        try:row=normalize(item)
        except ValueError as exc:
            decisions.append({'task_id':task,'reason':str(exc)});continue
        reason=None;match=None
        if task in reserved:reason='reserved_opencoder_holdout'
        elif item['seq_id'] in BROKEN_TESTS:reason='known_test_setup_uses_root_before_assignment'
        elif item['seq_id']==25886016399:reason='known_statement_example_dara_is_not_an_anagram_of_radar'
        key=normalized(row['instruction'])
        if reason is None:
            match=next((identifier for identifier,text in protected if same_statement(key,text)),None)
            if match is not None:reason='heldout_statement_overlap'
        if reason is None:
            match=next((identifier for identifier,text in seen if same_statement(key,text)),None)
            if match is not None:reason='duplicate_statement'
        if reason:
            decisions.append({'task_id':task,'reason':reason,**({'matches':match} if match is not None else {})})
        else:
            accepted.append(row);seen.append((task,key))
        if (i+1)%500==0:print(f'[prepare] {i+1}/{len(raw)} rows; training={len(accepted)}',flush=True)
    args.output.mkdir(parents=True)
    (args.output/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in accepted))
    (args.output/'validation.jsonl').write_bytes((args.evaluation/'development.jsonl').read_bytes())
    (args.output/'exclusions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in decisions))
    manifest={'protocol':'python-program-grpo-v1','source':imported,'model':MODEL,'revision':REVISION,
              'raw_rows':len(raw),'counts':{'train':len(accepted),'validation':len(read_rows(args.output/'validation.jsonl'))},
              'exclusions':dict(collections.Counter(r['reason'] for r in decisions)),
              'files_sha256':{s:sha(args.output/f'{s}.jsonl') for s in ['train','validation']},
              'evaluation_manifest':str(args.evaluation/'manifest.json'),'evaluation_manifest_sha256':sha(args.evaluation/'manifest.json'),
              'confirmation':evaluation['confirmation'],
              'training_reward':'All supplied Python test-program checks pass, binary reward. Source tests are not independently certified.',
              'development_protocol':'Existing original + expanded MBPP checks, unchanged.',
              'reference_solution_required':False,'difficulty_filter':False,
              'limitations':['Supplied tests may be weak or wrong. Passing them is not proof of correctness.','Statement matching removes exact/near-identical text, not every semantic duplicate or pretraining overlap.','Only plain Python test programs with explicit assert/raise checks are supported; pytest/unittest discovery is not integrated.']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({k:manifest[k] for k in ['raw_rows','counts','exclusions','reference_solution_required','difficulty_filter']},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,default=Path('data/opencoder-candidates/raw.jsonl'));p.add_argument('--output',type=Path,default=Path('data/python-programs'));p.add_argument('--evaluation',type=Path,default=Path('evaluation'))
    prepare(p.parse_args())
