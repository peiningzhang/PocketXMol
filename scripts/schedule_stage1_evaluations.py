"""Adopt live energy children and fill free CPU slots with disjoint evaluations."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
import pandas as pd
ROOT=Path('outputs_test_time_scaling/stage_1')

def process(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():raise RuntimeError(f'Adopted process vanished without exit evidence: {pid}')
    fields=path.read_text().split(') ',1)[1].split()
    return dict(state=fields[0],start_ticks=fields[19],exit_code=int(fields[49]))

def verify(folder):
    run=json.loads((folder/'run.json').read_text());out=folder/'tts_eval'
    info=pd.read_csv(folder/'gen_info.csv');metrics=pd.read_csv(out/'candidate_metrics.csv');energy=pd.read_csv(out/'energy_candidates.csv')
    names=set(info.filename)
    if len(energy)!=len(info) or set(energy.filename)!=names or set(metrics.filename)!=names:raise ValueError('Incomplete evaluation coverage')
    protocol=json.loads((out/'energy_protocol.json').read_text())
    if protocol['ensemble_threads']!=1 or protocol['ETKDG_randomSeed']!=42 or protocol['selection_uses_energy']:raise ValueError('Energy protocol differs')
    module=next(m for m in protocol['config']['modules'] if m['function']=='energy_ratio')
    if module['parameters']['ensemble_number_conformations']!=50 or module['parameters']['threshold_energy_ratio']!=100:raise ValueError('Energy configuration differs')
    if set(protocol['candidate_sha256'])!=names:raise ValueError('Cache provenance incomplete')
    for name in names:
        if hashlib.sha256((folder/'SDF'/name).read_bytes()).hexdigest()!=protocol['candidate_sha256'][name]:raise ValueError('Raw candidate changed')
    labels={f'best_of_{n}_{rank}' for n in [1,2,4,8,16] if n<=run['repeats'] for rank in ['random','confidence','geometry_confidence','vina_oracle']}
    summary=pd.read_csv(out/'energy_summary.csv').set_index('selection')
    if set(summary.index)!=labels|{'all_candidates'}:raise ValueError('Selection energy summary incomplete')
    for label in labels:
        if summary.loc[label,'n']!=run['num_pockets']:raise ValueError('Selection denominator differs')
    if summary.loc['all_candidates','n']!=len(info):raise ValueError('Candidate denominator differs')
    return len(info)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);a=p.parse_args()
    m=json.loads(a.manifest.read_text());slots=m['slots']
    name=m.get('name','evaluation_scheduler')
    lock=(ROOT/(name+'.lock')).open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    adopted=[];owned=[];history=[]
    for item in m['adopted']:
        state=process(item['pid']);cmd=Path(f"/proc/{item['pid']}/cmdline").read_bytes().decode().split('\0')
        if state['start_ticks']!=item['start_ticks']:raise ValueError('Adopted PID start time differs')
        if state['state']!='Z' and (item['folder'] not in cmd or 'scripts/evaluate_tts_energy.py' not in cmd):raise ValueError('Adopted PID identity differs')
        adopted.append(dict(item,start_ticks=state['start_ticks']))
    folders=[Path(x) for x in m['folders']];assert len(set(folders))==len(folders)
    if set(folders)&{Path(x['folder']) for x in adopted}:raise ValueError('Overlapping task ownership')
    def interrupt(signum,frame):raise KeyboardInterrupt('scheduler interrupted')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        while adopted or owned or folders:
            for item in list(adopted):
                state=process(item['pid'])
                if state['start_ticks']!=item['start_ticks']:raise ValueError('PID reused')
                if state['state']!='Z':continue
                if state['exit_code']!=0:raise RuntimeError('Adopted evaluator failed')
                folder=Path(item['folder']);count=verify(folder)
                (folder/'EVALUATED.recovery.json').write_text(json.dumps(dict(pid=item['pid'],exit_evidence=state,candidates=count,scope='Completed child retained by stopped queue parent'),indent=2))
                (folder/'EVALUATED').touch();history.append(dict(folder=str(folder),status='adopted_complete'));adopted.remove(item)
            for item in list(owned):
                code=item['process'].poll()
                if code is None:continue
                item['stream'].close()
                if code:raise RuntimeError(f"Evaluation failed: {item['folder']}")
                verify(item['folder']);history.append(dict(folder=str(item['folder']),status='complete'));owned.remove(item)
            while folders and len(adopted)+len(owned)<slots:
                folder=folders.pop(0)
                if (folder/'EVALUATED').exists():verify(folder);continue
                name=folder.parent.name+'_'+folder.name
                file=ROOT/(m.get('name','scheduler')+'_'+name+'.txt');file.write_text(str(folder)+'\n')
                stream=(ROOT/(m.get('name','scheduler')+'_'+name+'.log')).open('w')
                cmd=[sys.executable,'scripts/evaluate_stage1.py','--root',str(folder.parent),'--folders_file',str(file),'--vina_workers','1']
                child=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                owned.append(dict(folder=folder,process=child,stream=stream))
            (ROOT/(m.get('name','evaluation_scheduler')+'.json')).write_text(json.dumps(dict(active_adopted=adopted,active_owned=[dict(folder=str(x['folder']),pid=x['process'].pid) for x in owned],pending=list(map(str,folders)),completed=history,slots=slots),indent=2))
            if adopted or owned or folders:time.sleep(10)
        (ROOT/(m.get('complete_marker','EVALUATION_SCHEDULER_COMPLETE'))).touch()
    finally:
        for item in owned:
            if item['process'].poll() is None:
                os.killpg(item['process'].pid,signal.SIGTERM);item['process'].wait()
            item['stream'].close()
        if m.get('resume_pid') is not None:
            subprocess.run(['kill','-CONT',str(m['resume_pid'])],check=True)
        (ROOT/(m.get('name','evaluation_scheduler')+'_resumed.json')).write_text(json.dumps(dict(pid=m.get('resume_pid'),timestamp=time.time()),indent=2))
if __name__=='__main__':main()
