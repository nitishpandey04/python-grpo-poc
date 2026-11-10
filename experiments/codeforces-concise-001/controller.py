"""One-off two-stage experiment using the existing training entry point."""
import hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
PLAN=json.loads((WORK/'plan.json').read_text())
ENV=dict(os.environ,HF_HOME=str(ROOT/'.cache/huggingface'),TRACKIO_DIR=str(ROOT/'runs/trackio'),TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
PYTHON=str(ROOT/'.venv/bin/python')

def read(path):return json.loads(Path(path).read_text())
def status(phase,**values):
    row={'phase':phase,'updated_at':time.time(),**values}
    tmp=WORK/'queue-status.tmp';tmp.write_text(json.dumps(row,indent=2));tmp.replace(WORK/'queue-status.json')
    print(json.dumps(row),flush=True)

def check_frozen_source():
    for name,digest in read(WORK/'source-sha256.json').items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,('Source changed since plan',name)

def run(name,extra):
    check_frozen_source()
    folder=ROOT/'runs'/name
    if folder.exists():raise RuntimeError('Run directory already exists: '+name)
    folder.mkdir()
    command=[PYTHON,'-u',str(ROOT/'train.py'),'--name',name,'--model',PLAN['model'],'--model-revision',PLAN['model_revision'],'--load-in-4bit','--seed','42','--max-tokens','1024',*extra]
    status('running',run=name)
    with (folder/'console.log').open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=ENV,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        (folder/'process.pid').write_text(str(process.pid))
        code=process.wait()
    if code or read(folder/'status.json')['phase']!='complete':raise RuntimeError(f'{name} failed, exit code {code}; inspect its saved log')
    config=read(folder/'config.json')
    assert config['dataset_manifest_sha256']==PLAN['dataset_manifest_sha256']
    return folder,read(folder/'report.json')

def main():
    status('starting probe')
    folder,probe=run(PLAN['new_probe'],['--reward-probe','--probe-problems','32','--concise-code'])
    assert read(folder/'config.json')['probe_selection_sha256']==PLAN['probe_selection_sha256']
    old=read(WORK/'original-probe-report.json')
    concise=probe['overall']['correct_candidates']>=old['overall']['correct_candidates'] and probe['overall']['token_cap_hits']<old['overall']['token_cap_hits']
    decision={'selected_prompt':'concise' if concise else 'original','rule':PLAN['selection_rule'],'original':old['overall'],'concise':probe['overall'],'by_difficulty':{'original':old['by_difficulty'],'concise':probe['by_difficulty']},'development_used_for_prompt_selection':False}
    (WORK/'prompt-decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    status('prompt selected',selected_prompt=decision['selected_prompt'])
    if shutil.disk_usage(ROOT).free<PLAN['minimum_free_disk_bytes_before_training']:raise RuntimeError('Insufficient free disk for training checkpoints; no user data deleted')
    folder,report=run(PLAN['training_run'],['--steps','50','--eval-every','10',*(['--concise-code'] if concise else [])])
    assert report['steps']==50 and not report['confirmation_evaluated']
    results={'prompt_decision':decision,'training_report':report,'training_run':PLAN['training_run']}
    (WORK/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    status('complete',run=PLAN['training_run'],selected_prompt=decision['selected_prompt'],selected_checkpoint=report['selected_checkpoint'])

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('failed',error=str(exc));traceback.print_exc();raise
