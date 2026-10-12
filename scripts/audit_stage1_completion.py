"""Evidence audit; missing cells never count as a completed experiment."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_stage1 import tasks
from scripts.stage1_resources import charged_hours
import pandas as pd
ROOT=Path('outputs_test_time_scaling/stage_1')
def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());checks=[]
    def check(name,passed,details):checks.append(dict(requirement=name,passed=bool(passed),details=details))
    ledger=json.loads((ROOT/'generation_ledger.json').read_text())
    planned=list(tasks());authoritative={r['label']:r for r in ledger if r['status']=='done'}
    missing=[];issues=[];candidates=0;checkpoint_hashes=set()
    for task in planned:
        rec=authoritative.get(task['label'])
        if rec is None:missing.append(task['label']);continue
        folder=Path(rec['outdir'])
        if not (folder/'DONE').exists():issues.append(f'{task["label"]}: no DONE');continue
        run=json.loads((folder/'run.json').read_text());checkpoint_hashes.add(run['checkpoint_sha256'])
        f=pd.read_csv(folder/'gen_info.csv');candidates+=len(f)
        repeats=range(4,8) if task['kind']=='pool_tail' else range(task['repeats'])
        expected={(pid,i) for pid in protocol['pocket_ids'] for i in repeats}
        observed=set(zip(f.data_id,f.i_repeat))
        if observed!=expected or len(f)!=len(expected):issues.append(f'{task["label"]}: candidate grid differs')
        evaluated=folder
        if task['kind']=='pool_tail':evaluated=ROOT/f"pool_t100_n8_r0_s{task['seed']}"
        if not (evaluated/'EVALUATED').exists():issues.append(f'{task["label"]}: evaluation incomplete')
        else:
            metrics=pd.read_csv(evaluated/'tts_eval/candidate_metrics.csv');energy=pd.read_csv(evaluated/'tts_eval/energy_candidates.csv')
            if set(metrics.filename)!=set(energy.filename):issues.append(f'{task["label"]}: energy coverage records incomplete')
            ep=json.loads((evaluated/'tts_eval/energy_protocol.json').read_text())
            if ep.get('ensemble_threads')!=1 or ep.get('ETKDG_randomSeed')!=42:issues.append(f'{task["label"]}: energy protocol differs')
    check('core_40_tasks_7000_candidates_and_evaluation',not missing and not issues and candidates==7000,dict(missing=missing,issues=issues,planned_tasks=len(planned),observed_candidates=candidates))
    checkpoint=Path('data/trained_models/pxm/checkpoints/pocketxmol.ckpt')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    check('teacher_checkpoint_unchanged',checkpoint_hashes=={digest},dict(recorded=list(checkpoint_hashes),current=digest))
    manifest=ROOT/'confirmation/confirmation_manifest.json'
    check('immutable_confirmation_selection',manifest.exists(),str(manifest))
    if manifest.exists():
        frozen=json.loads(manifest.read_text())
        changed=[file for file,digest in frozen['code_sha256'].items()
                 if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest]
        check('frozen_implementation_unchanged',not changed,changed)
        confirmation=ROOT/'confirmation'
        path=confirmation/'confirmation_ledger.json'
        records=json.loads(path.read_text()) if path.exists() else []
        done={r['label']:r for r in records if r['status']=='done'}
        expected={f"{c['name']}_b{b}_s{s}" for c in frozen['selected']
                  for b in frozen['budgets'] for s in frozen['seeds']}
        problems=[]
        for label in sorted(expected):
            rec=done.get(label)
            if rec is None:problems.append(label+': generation missing');continue
            folder=Path(rec['outdir'])
            if not (folder/'EVALUATED').exists():problems.append(label+': evaluation missing');continue
            info=pd.read_csv(folder/'gen_info.csv')
            run=json.loads((folder/'run.json').read_text())
            grid={(pid,i) for pid in protocol['confirmation_ids'] for i in range(run['repeats'])}
            if set(zip(info.data_id,info.i_repeat))!=grid or len(info)!=len(grid):
                problems.append(label+': candidate grid differs')
            metrics=pd.read_csv(folder/'tts_eval/candidate_metrics.csv')
            energy=pd.read_csv(folder/'tts_eval/energy_candidates.csv')
            if set(metrics.filename)!=set(info.filename) or set(energy.filename)!=set(info.filename):
                problems.append(label+': candidate evaluation coverage differs')
            energy_protocol=json.loads((folder/'tts_eval/energy_protocol.json').read_text())
            if energy_protocol.get('ensemble_threads')!=1 or energy_protocol.get('ETKDG_randomSeed')!=42:
                problems.append(label+': energy protocol differs')
            if run['checkpoint_sha256']!=digest:problems.append(label+': checkpoint differs')
        check('confirmation_full_matrix',not problems and set(done)==expected,problems)
    nested=ROOT/'nested_pool_audit.json'
    nested_data=json.loads(nested.read_text()) if nested.exists() else {}
    check('nested_pool_raw_and_selection_identity',nested_data.get('complete') is True,nested_data)
    analysis=ROOT/'tables/analysis.json';data=json.loads(analysis.read_text()) if analysis.exists() else {}
    check('full_statistics_matrix',bool(data) and data.get('provisional') is False,data)
    check('pocket_cluster_bootstrap_at_least_5000',data.get('bootstrap_resamples',0)>=5000
          and data.get('cluster')=='pocket after seed average',
          dict(resamples=data.get('bootstrap_resamples'),cluster=data.get('cluster')))
    decision_path=ROOT/'tables/n16_decision.json'
    decision=json.loads(decision_path.read_text()) if decision_path.exists() else {}
    terminal=decision.get('baseline_complete') is True and decision.get('decision') in [
        'bounded_N16_extension_warranted','no_clear_B800_upward_trend_no_N16_expansion']
    check('conditional_N16_decision_after_complete_core',terminal,decision)
    if decision.get('decision')=='bounded_N16_extension_warranted':
        extension=ROOT/'n16_extension';stats=extension/'tables/analysis.json'
        evidence=json.loads(stats.read_text()) if stats.exists() else {}
        artifacts=['EVALUATION_COMPLETE','tables/paired_comparisons.csv','tables/paired_budget_differences.csv',
                   'figures/actual_nfe_curves.pdf','figures/logical_search_time_curves.pdf']
        check('triggered_N16_extension_evaluated_and_reported',evidence.get('complete') is True
              and all((extension/file).exists() for file in artifacts),evidence)
    for name in ['timing','timing_trace_control']:
        folder=ROOT/name;p=folder/'timing_ledger.json';records=json.loads(p.read_text()) if p.exists() else []
        expected_timing={(method,budget,trial) for method in
                         ['geometry_confidence','more_steps','native_refinement']+
                         ([c['name'] for c in frozen['selected']] if manifest.exists() else [])
                         for budget in [400,800] for trial in range(3)}
        completed={(r['method'],r['budget'],r['trial']) for r in records if r['status']=='done'}
        check(name+'_three_repeats_complete',(folder/'DONE').exists() and completed==expected_timing
              and len(completed)==30 and all(r['status']=='done' for r in records),
              dict(done=sum(r['status']=='done' for r in records),marker=str(folder/'DONE')))
    for artifact in ['report.md','method_notes.md','tables/paired_comparisons.csv','tables/paired_budget_differences.csv','tables/n16_decision.json','tables/pool_oracle.csv','tables/oracle_gap.csv','tables/pool_diversity.csv','tables/confidence_correlations.csv','tables/confidence_reliability.csv']:
        p=ROOT/artifact;check('artifact:'+artifact,p.exists() and p.stat().st_size>0,str(p))
    figures=list((ROOT/'figures').glob('*.pdf'));check('exportable_figures',bool(figures),[str(p) for p in figures])
    matched_path=ROOT/'tables/native_actual_nfe_matched_protocol.json'
    matched=json.loads(matched_path.read_text()) if matched_path.exists() else {}
    check('native_exact_actual_NFE_diagnostic',matched.get('complete') is True
          and matched.get('pairs')==1500 and matched.get('all_planned_native_outputs_retained') is True,matched)
    prior_path=ROOT/'tables/atom_count_strategy_audit.json'
    prior=json.loads(prior_path.read_text()) if prior_path.exists() else {}
    check('FK_atom_count_prior_fairness',prior.get('complete') is True,prior)
    controlled=ROOT/'figures/confirmation_controlled_time_curves.pdf'
    check('confirmation_quality_with_repeated_controlled_cost',controlled.exists(),str(controlled))
    gates_path=ROOT/'tables/stage2_quantitative_gates.json'
    check('stage2_quantitative_decision_evidence',gates_path.exists(),str(gates_path))
    hours=charged_hours();check('48_gpu_hour_limit',hours<=48,hours)
    result=dict(complete=all(c['passed'] for c in checks),checks=checks,scope='Automated necessary checks; report/figure interpretation and Stage2 guardrails additionally require final manual evidence review')
    (ROOT/'completion_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(complete=result['complete'],passed=sum(c['passed'] for c in checks),total=len(checks),unmet=[c['requirement'] for c in checks if not c['passed']]),indent=2))
if __name__=='__main__':main()
