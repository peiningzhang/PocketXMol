"""Frozen teacher native-chain FK particles; P=4/8, actual graph-NFE accounting."""
import argparse
import copy
import hashlib
import json
import sys
import time
MODULE_STARTED = time.perf_counter()
from pathlib import Path
SAMPLER_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from torch_geometric.data import Batch
from scripts.eval_consistency import _build_test_loader,_load_student,_post_process_batch
from scripts.fk_particles import fk_loop
from models.maskfill import PMAsymDenoiser
from models.sample import get_cfd_traj,seperate_outputs2
from utils.consistency import infer_train_config_path_from_ckpt
from utils.misc import make_config,seed_all,save_config,get_logger
from utils.sample_noise import get_sample_noiser


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    main_started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--pocket_ids',type=Path)
    p.add_argument('--num_pockets',type=int,default=3)
    p.add_argument('--particles',type=int,choices=[4,8],required=True)
    p.add_argument('--interval',type=int,choices=[20,50],required=True)
    p.add_argument('--strength',type=float,choices=[2.,5.],required=True)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--batch_size',type=int,default=4)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--checkpoint',default='data/trained_models/pxm/checkpoints/pocketxmol.ckpt')
    p.add_argument('--config_task',default='configs/sample/test/sbdd_csd/simple.yml')
    p.add_argument('--warmup',action='store_true')
    a=p.parse_args()
    from scripts.stage1_resources import check_budget
    check_budget(a.outdir)
    root=a.outdir;root.mkdir(parents=True,exist_ok=False);(root/'SDF').mkdir()
    logger=get_logger('fk_sample',str(root)); task=make_config(a.config_task);task.noise.num_steps=a.steps
    task.sample.seed=a.seed;task.sample.batch_size=a.batch_size;task.sample.num_repeats=a.particles;save_config(task,str(root/'sample.yml'))
    train=make_config(infer_train_config_path_from_ckpt(a.checkpoint));seed_all(a.seed)
    data_tick=time.perf_counter()
    loader,featurizer,dims=_build_test_loader(train,task,a.batch_size,0);dataset=loader.dataset
    lookup={str(dataset[i].data_id):i for i in range(len(dataset))}
    ids=a.pocket_ids.read_text().splitlines() if a.pocket_ids else list(lookup)[:a.num_pockets]
    if len(set(ids))!=len(ids) or any(i not in lookup for i in ids):raise ValueError('Invalid pocket IDs')
    data_setup_seconds=time.perf_counter()-data_tick
    model_tick=time.perf_counter()
    model=PMAsymDenoiser(config=train.model,**dims).to('cuda:0');_load_student(model,a.checkpoint,use_ema=False,map_location='cuda:0');model.eval()
    noiser=get_sample_noiser(task.noise,dims['num_node_types'],dims['num_edge_types'],mode='sample',device='cuda:0',ref_config=train.noise)
    model_noiser_setup_seconds=time.perf_counter()-model_tick
    counts={'forward_calls':0,'graph_forward_calls':0}
    def hook(module,inputs,outputs):
        counts['forward_calls']+=1;counts['graph_forward_calls']+=inputs[0].num_graphs
    def population(pid):
        graphs=[]
        for repeat in range(a.particles):
            seed_all(a.seed+repeat)
            data=dataset[lookup[pid]]
            data['i_repeat']=torch.tensor([repeat]);graphs.append(data)
        seed_all(a.seed+lookup[pid]*1009)
        return Batch.from_data_list(graphs,follow_batch=loader.follow_batch,exclude_keys=loader.exclude_keys).to('cuda:0')
    warmup_seconds=0.
    warmup_calls={"forward_calls":0,"graph_forward_calls":0}
    def warm_hook(module,inputs,outputs):
        warmup_calls["forward_calls"]+=1;warmup_calls["graph_forward_calls"]+=inputs[0].num_graphs
    if a.warmup:
        warm_handle=model.register_forward_hook(warm_hook)
        tick=time.perf_counter()
        fk_loop(population(ids[0]),model,copy.deepcopy(noiser),interval=a.interval,strength=a.strength,
                follow_batch=loader.follow_batch,diagnostics=[],max_batch=a.batch_size)
        torch.cuda.synchronize();warmup_seconds=time.perf_counter()-tick;warm_handle.remove()
    handle=model.register_forward_hook(hook);rows=[];saved=0;sample_seconds=0.;verifier_seconds=0.;diagnostics=[];batch_costs=[]
    started=time.perf_counter()
    data_loading_seconds=0.;reconstruction_seconds=0.
    for pid in ids:
        check_budget(a.outdir)
        before=dict(counts);data_tick=time.perf_counter();batch=population(pid);torch.cuda.synchronize()
        data_loading_seconds+=time.perf_counter()-data_tick;tick=time.perf_counter()
        batch,outputs,verifier=fk_loop(batch,model,noiser,interval=a.interval,strength=a.strength,
            follow_batch=loader.follow_batch,diagnostics=diagnostics,max_batch=a.batch_size)
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick;sample_seconds+=elapsed;verifier_seconds+=verifier
        # Resampling copies repeat metadata; final slots are planned candidate identities.
        batch['i_repeat']=torch.arange(a.particles,device='cuda:0')
        reconstruction_tick=time.perf_counter()
        new_rows,saved=_post_process_batch(batch,outputs,featurizer,str(root/'SDF'),saved,logger)
        reconstruction_seconds+=time.perf_counter()-reconstruction_tick
        _,split,_=seperate_outputs2(batch,outputs,None,off_tqdm=True)
        for row,out in zip(new_rows,split):
            row['nfe']=a.steps;row['cfd_traj']=float(get_cfd_traj(out['confidence_pos_traj'],steps=a.steps))
        rows.extend(new_rows);pd.DataFrame(rows).to_csv(root/'gen_info.csv',index=False)
        batch_costs.append(dict(data_ids=[pid],seconds=elapsed,graph_nfe=counts['graph_forward_calls']-before['graph_forward_calls']))
        (root/'particle_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
        logger.info('pocket=%s outputs=%d NFE=%d',pid,saved,counts['graph_forward_calls'])
    handle.remove()
    manifest={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    manifest.update(**counts,pocket_ids=ids,num_pockets=len(ids),requested_num_pockets=a.num_pockets,num_generated=saved,repeats=a.particles,start_repeat=0,
        timing_version=3,import_seconds=main_started-MODULE_STARTED,startup_seconds=started-main_started-warmup_seconds,
        data_setup_seconds=data_setup_seconds,model_noiser_setup_seconds=model_noiser_setup_seconds,
        data_loader_and_stream_setup_seconds=data_loading_seconds,reconstruction_seconds=reconstruction_seconds,
        actual_nfe_per_candidate=counts['graph_forward_calls']/saved,sampling_seconds=sample_seconds,
        intermediate_verifier_seconds=verifier_seconds,warmup_seconds=warmup_seconds,warmup_calls=warmup_calls,
        generation_wall_seconds=time.perf_counter()-started,batch_costs=batch_costs,
        checkpoint_sha256=sha256(a.checkpoint),config_sha256=sha256(root/'sample.yml'),
        sampler_sha256=SAMPLER_SOURCE_SHA256_AT_IMPORT,fk_implementation_sha256=sha256('scripts/fk_particles.py'),
        torch_version=torch.__version__,gpu_name=torch.cuda.get_device_name(),method='fk_particles')
    from scripts.stage1_runtime import runtime_metadata
    manifest['runtime']=runtime_metadata()
    (root/'run.json').write_text(json.dumps(manifest,indent=2));(root/'pocket_ids.txt').write_text('\n'.join(ids)+'\n');(root/'DONE').touch()

if __name__=='__main__':main()
