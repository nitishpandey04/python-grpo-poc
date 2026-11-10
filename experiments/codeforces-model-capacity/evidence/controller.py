"""One-off experiment queue; all inference/training uses the existing train.py."""
import hashlib,json,os,shutil,signal,subprocess,sys,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/model-capacity-comparison'
SOURCES=json.loads((OUT/'sources.json').read_text())
ENV=dict(os.environ,HF_HOME=str(ROOT/'.cache/huggingface'),TRACKIO_DIR=str(ROOT/'runs/trackio'),TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',HF_HUB_DISABLE_XET='1',HF_HUB_DOWNLOAD_TIMEOUT='60')
PYTHON=str(ROOT/'.venv/bin/python')

def status(phase,**values):
    row={'phase':phase,'updated_at':time.time(),**values}
    (OUT/'queue-status.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)

def repo_dir(source):return ROOT/'.cache/huggingface/hub'/('models--'+source['repo'].replace('/','--'))
def snapshot(source):return repo_dir(source)/'snapshots'/source['revision']
def verified_weights(source):
    for file in source['weights']:
        path=snapshot(source)/file['path']
        if not path.exists():return False
        if path.stat().st_size!=file['bytes']:raise RuntimeError('Unexpected weight size: '+str(path))
        with path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=file['lfs_sha256']:raise RuntimeError('Weight checksum mismatch: '+str(path))
    return True

def download(source):
    status('downloading',model=source['repo'])
    cmd=['/home/nitish/.local/bin/hf','download',source['repo'],'--revision',source['revision'],'--cache-dir',str(ROOT/'.cache/huggingface/hub')]
    for pattern in ['*.safetensors','*.json','*.txt','*.jinja','README.md','LICENSE*']:cmd+=['--include',pattern]
    for attempt in range(3):
        try:
            subprocess.run(cmd,cwd=ROOT,env=ENV,check=True);break
        except subprocess.CalledProcessError:
            if attempt==2:raise
            status('download retry',model=source['repo'],attempt=attempt+1);time.sleep(10)
    if not verified_weights(source):raise RuntimeError('Weights missing after CLI download')
    status('weights verified',model=source['repo'])

def run(source,name,probe=False,quant=False,micro=2):
    folder=ROOT/'runs'/name
    old=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    if old.get('phase')=='complete':return json.loads((folder/'report.json').read_text())
    if (folder/'failure.json').exists():return None
    if (folder/'config.json').exists():raise RuntimeError('Unfinished run needs inspection: '+name)
    args=['--name',name,'--model',source['repo'],'--model-revision',source['revision'],'--micro-batch',str(micro)]
    args+=['--reward-probe'] if probe else ['--steps','10','--eval-every','10']
    if quant:args+=['--load-in-4bit']
    status('starting run',run=name,model=source['repo'],quantized=quant,micro_batch=micro)
    subprocess.run([PYTHON,'launch.py','train',*args],cwd=ROOT,env=ENV,check=True)
    pid=int((folder/'process.pid').read_text())
    while True:
        state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
        if state.get('phase')=='complete':return json.loads((folder/'report.json').read_text())
        if state.get('phase','').startswith('failed'):
            if (folder/'failure.json').exists():return None
            raise RuntimeError('Run failed; inspect console: '+name)
        proc=Path(f'/proc/{pid}/stat')
        if not proc.exists() or proc.read_text().split()[2]=='Z':
            if (folder/'failure.json').exists():return None
            raise RuntimeError('Run exited without a completed report: '+name)
        status('run active',run=name,run_phase=state.get('phase','starting'),optimizer_step=state.get('optimizer_step',0))
        time.sleep(15)

def clear_weights(source):
    actions=json.loads((OUT/'cache-actions.json').read_text()) if (OUT/'cache-actions.json').exists() else []
    for file in source['weights']:
        digest=file.get('lfs_sha256') or file['sha256']
        blob=repo_dir(source)/'blobs'/digest;link=snapshot(source)/file['path']
        if blob.exists():
            assert blob.stat().st_size==file['bytes']
            if link.is_symlink():
                assert link.resolve()==blob.resolve();link.unlink()
            blob.unlink()
            actions.append({'model':source['repo'],'revision':source['revision'],'path':file['path'],'sha256':digest,'bytes':file['bytes'],'reason':'Redownloadable cache evicted for sequential model comparison','time':time.time()})
    (OUT/'cache-actions.json').write_text(json.dumps(actions,indent=2))

def ensure_space(source):
    required=sum(f['bytes'] for f in source['weights'])+int(1.5*1024**3)
    if shutil.disk_usage(ROOT).free<required:
        old=json.loads((ROOT/'experiments/codeforces-1.5b-001/model-source.json').read_text())
        clear_weights(old)
    if shutil.disk_usage(ROOT).free<required:raise RuntimeError('Insufficient disk space after bounded weight-cache cleanup')

def dashboard():
    try:
        with urllib.request.urlopen('http://127.0.0.1:7860/?project=python-grpo',timeout=5) as r:
            if r.status==200:return
    except OSError:pass
    pidfile=ROOT/'runs/dashboard/process.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text());p=Path(f'/proc/{pid}/cmdline')
        if p.exists() and b'trackio.show' in p.read_bytes():os.kill(pid,signal.SIGTERM);time.sleep(2)
    subprocess.run([PYTHON,'launch.py','dashboard'],cwd=ROOT,env=ENV,check=True)


def main():
    dashboard();results=json.loads((OUT/'results.json').read_text()) if (OUT/'results.json').exists() else []
    completed={r['source']['repo'] for r in results if r.get('selected_training_run')}
    # The first CLI download was started interactively; reuse its cache when ready.
    deadline=time.monotonic()+3600
    while SOURCES[0]['repo'] not in completed and not all((snapshot(SOURCES[0])/f['path']).exists() for f in SOURCES[0]['weights']):
        status('waiting for initial CLI download',model=SOURCES[0]['repo'])
        if time.monotonic()>deadline:raise RuntimeError('Initial CLI download did not finish')
        time.sleep(15)
    if SOURCES[0]['repo'] not in completed and not verified_weights(SOURCES[0]):raise RuntimeError('First weight verification failed')
    for index,source in enumerate(SOURCES):
        if source['repo'] in completed:continue
        label='3b' if index==0 else '4b'
        if index:
            clear_weights(SOURCES[index-1]);ensure_space(source);download(source)
        probe_name=f'codeforces-{label}-probe-bf16-001'
        pr=run(source,probe_name,probe=True)
        entry={'source':source,'attempts':[],'probe':probe_name,'selected_training_run':None}
        if pr is not None:
            name=f'codeforces-{label}-smoke-bf16-001';entry['attempts'].append(name)
            sr=run(source,name)
            if sr is None:
                fail=json.loads((ROOT/'runs'/name/'failure.json').read_text())
                if fail['phase']=='updating weights':
                    name=f'codeforces-{label}-smoke-bf16-micro1-001';entry['attempts'].append(name);sr=run(source,name,micro=1)
            if sr is not None:entry['selected_training_run']=name
        if entry['selected_training_run'] is None:
            probe_name=f'codeforces-{label}-probe-nf4-001';entry['quantized_probe']=probe_name
            pr=run(source,probe_name,probe=True,quant=True)
            if pr is not None:
                for micro in [2,1]:
                    name=f'codeforces-{label}-smoke-nf4-micro{micro}-001';entry['attempts'].append(name)
                    sr=run(source,name,quant=True,micro=micro)
                    if sr is not None:entry['selected_training_run']=name;break
                    fail=json.loads((ROOT/'runs'/name/'failure.json').read_text())
                    if fail['phase']!='updating weights':break
        results.append(entry);(OUT/'results.json').write_text(json.dumps(results,indent=2))
    status('complete',models=results)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('failed',error_type=type(exc).__name__,message=str(exc));raise
