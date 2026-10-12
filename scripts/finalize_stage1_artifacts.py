"""Wait for formal evaluation completion, then render analysis artifacts.

This never marks the user goal complete: final evidence/figure review is manual.
"""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path('outputs_test_time_scaling/stage_1')
PYTHON='/shared/healthinfolab/phz24002/anaconda3/envs/reed_molecule/bin/python'
def main():
    with (ROOT/'finalization.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((ROOT/'confirmation/confirmation_manifest.json').read_text())
        expected=[ROOT/'confirmation'/f"{c['name']}_b{b}_s{s}" for c in manifest['selected'] for b in manifest['budgets'] for s in manifest['seeds']]
        # Selected development extensions and all core are already independently audited.
        while True:
            missing=[str(p) for p in expected if not (p/'EVALUATED').exists()]
            (ROOT/'finalization_status.json').write_text(json.dumps(dict(phase='waiting_evaluation',missing=missing,timestamp=time.time()),indent=2))
            if not missing:break
            time.sleep(20)
        commands=[['snapshot_stage1_progress.py'],['analyze_stage1.py','--resamples','5000'],
            ['analyze_stage1_budget_trends.py'],['analyze_stage1_native_matched.py'],
            ['analyze_stage1_guardrails.py'],['analyze_stage1_timing.py'],
            ['analyze_stage1_timing.py','--trace_control'],['analyze_stage1_trace_costs.py'],
            ['analyze_stage1_stage2_gates.py'],['plot_stage1.py'],
            ['plot_stage1_controlled_timing.py'],['write_stage1_report.py'],['audit_stage1_completion.py']]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/tmp/pxm_stage1_mpl')
        try:
            for args in commands:
                (ROOT/'finalization_status.json').write_text(json.dumps(dict(phase='running',command=args,timestamp=time.time()),indent=2))
                with (ROOT/('finalization_'+Path(args[0]).stem+('_trace_control' if '--trace_control' in args else '')+'.log')).open('w') as stream:
                    subprocess.run([PYTHON,'scripts/'+args[0],*args[1:]],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            (ROOT/'ARTIFACTS_RENDERED_PENDING_MANUAL_REVIEW').touch()
            (ROOT/'finalization_status.json').write_text(json.dumps(dict(phase='artifacts_rendered_pending_manual_review',timestamp=time.time()),indent=2))
        except Exception as e:
            (ROOT/'finalization_error.json').write_text(json.dumps(dict(error=repr(e),timestamp=time.time()),indent=2));raise
if __name__=='__main__':main()
