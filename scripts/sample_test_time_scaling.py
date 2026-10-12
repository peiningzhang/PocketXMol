"""Frozen PocketXMol baselines with actual forward-call accounting.

Uses the native sampler: more steps, more independent candidates,
and PocketXMol's existing confidence-guided partial refinement (base.yml).
"""
import argparse
import copy
import json
import os
import sys
import time
MODULE_STARTED = time.perf_counter()
import hashlib
import subprocess
from pathlib import Path
SAMPLER_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from scripts.eval_consistency import _build_test_loader, _load_student, _post_process_batch
from models.maskfill import PMAsymDenoiser
from models.sample import sample_loop3, get_cfd_traj
from utils.consistency import infer_train_config_path_from_ckpt
from utils.misc import make_config, seed_all, save_config, get_logger
from utils.sample_noise import get_sample_noiser


def main():
    main_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config_task', default='configs/sample/test/sbdd_csd/simple.yml')
    parser.add_argument('--checkpoint', default='data/trained_models/pxm/checkpoints/pocketxmol.ckpt')
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--start_repeat', type=int, default=0)
    parser.add_argument('--num_pockets', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max_refine_rounds', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pocket_ids', help='Ordered newline-separated pocket IDs')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--no_full_trajectory', action='store_true')
    args = parser.parse_args()
    from scripts.stage1_resources import check_budget
    check_budget(args.outdir)
    if min(args.steps, args.repeats, args.num_pockets, args.batch_size) < 1:
        parser.error('steps, repeats, num_pockets and batch_size must be positive')
    if args.no_full_trajectory:
        from functools import partial
        sample_fn = partial(sample_loop3, record_trajectory=False)
    else:
        sample_fn = sample_loop3
    root = Path(args.outdir)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'SDF').mkdir()
    logger = get_logger('tts_sample', str(root))
    task = make_config(args.config_task)
    task.noise.num_steps = args.steps
    is_ar = task.task.transform.name
    if is_ar.startswith('ar'):
        task.noise.ar_config.max_ar_step = args.max_refine_rounds
    task.sample.num_repeats = args.repeats
    task.sample.seed = args.seed
    task.sample.batch_size = args.batch_size
    save_config(task, str(root / 'sample.yml'))
    train = make_config(infer_train_config_path_from_ckpt(args.checkpoint))
    seed_all(args.seed)
    data_tick = time.perf_counter()
    loader, featurizer, dims = _build_test_loader(train, task, args.batch_size, 0)
    dataset = loader.dataset
    indices = list(range(min(args.num_pockets, len(dataset))))
    if args.pocket_ids:
        wanted = Path(args.pocket_ids).read_text().splitlines()
        lookup = {str(dataset[i]['data_id']): i for i in range(len(dataset))}
        if len(set(wanted)) != len(wanted) or any(x not in lookup for x in wanted):
            raise ValueError('Duplicate or unknown pocket IDs')
        indices = [lookup[x] for x in wanted]
    pocket_ids = [str(dataset[i]['data_id']) for i in indices]
    (root / 'pocket_ids.txt').write_text('\n'.join(pocket_ids) + '\n')
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        follow_batch=loader.follow_batch, exclude_keys=loader.exclude_keys)
    data_setup_seconds = time.perf_counter() - data_tick
    model_tick = time.perf_counter()
    model = PMAsymDenoiser(config=train.model, **dims).to(args.device)
    _load_student(model, args.checkpoint, use_ema=False, map_location=args.device)
    model.eval()
    noiser = get_sample_noiser(task.noise, dims['num_node_types'], dims['num_edge_types'],
                              mode='sample', device=args.device, ref_config=train.noise)
    model_noiser_setup_seconds = time.perf_counter() - model_tick
    warmup_seconds = 0.0
    warmup_calls = {"forward_calls": 0, "graph_forward_calls": 0}
    def count_warmup(module, inputs, output):
        warmup_calls["forward_calls"] += 1
        warmup_calls["graph_forward_calls"] += inputs[0].num_graphs
    if args.warmup:
        warmup_handle = model.register_forward_hook(count_warmup)
        tick = time.perf_counter()
        warm_batch = next(iter(loader)).to(args.device)
        sample_fn(warm_batch, model, copy.deepcopy(noiser), args.device,
                     is_ar=is_ar, off_tqdm=True)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        warmup_seconds = time.perf_counter() - tick
        warmup_handle.remove()
        del warm_batch
    calls = {'forward_calls': 0, 'graph_forward_calls': 0}
    def count_call(module, inputs, output):
        calls['forward_calls'] += 1
        calls['graph_forward_calls'] += inputs[0].num_graphs
    handle = model.register_forward_hook(count_call)
    rows, batch_costs = [], []
    saved = 0
    sample_seconds = 0.0
    started = time.perf_counter()
    previous_batch_end = started
    data_loader_seconds = 0.0
    reconstruction_seconds = 0.0
    for repeat in range(args.start_repeat, args.start_repeat + args.repeats):
        # Each repeat has a stable independent stream shared across methods.
        seed_all(args.seed + repeat)
        for batch in loader:
            data_loader_seconds += time.perf_counter() - previous_batch_end
            check_budget(args.outdir)
            batch = batch.to(args.device)
            batch['i_repeat'] = torch.full((batch.num_graphs,), repeat, device=args.device)
            before = copy.copy(calls)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            tick = time.perf_counter()
            batch, outputs, _ = sample_fn(batch, model, noiser, args.device,
                                            is_ar=is_ar, off_tqdm=True)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - tick
            sample_seconds += elapsed
            reconstruction_tick = time.perf_counter()
            new_rows, saved = _post_process_batch(batch, outputs, featurizer,
                                                  str(root / 'SDF'), saved, logger)
            reconstruction_seconds += time.perf_counter() - reconstruction_tick
            from models.sample import seperate_outputs2
            _, split_outputs, _ = seperate_outputs2(batch, outputs, None, off_tqdm=True)
            for row, out in zip(new_rows, split_outputs):
                row['cfd_traj'] = float(get_cfd_traj(out['confidence_pos_traj'], steps=args.steps))
                row['nfe'] = calls['forward_calls'] - before['forward_calls']
            rows.extend(new_rows)
            batch_costs.append({'repeat': repeat, 'data_ids': [r['data_id'] for r in new_rows],
                                'seconds': elapsed, 'forward_calls': calls['forward_calls'] - before['forward_calls']})
            pd.DataFrame(rows).to_csv(root / 'gen_info.csv', index=False)
            logger.info('repeat=%d samples=%d sample_seconds=%.2f calls=%s', repeat, saved, sample_seconds, calls)
            del batch, outputs, split_outputs
            previous_batch_end = time.perf_counter()
    handle.remove()
    manifest = dict(vars(args), **calls, num_generated=saved,
                    timing_version=3, import_seconds=main_started-MODULE_STARTED,
                    startup_seconds=started-main_started-warmup_seconds,
                    data_setup_seconds=data_setup_seconds, model_noiser_setup_seconds=model_noiser_setup_seconds,
                    data_loader_and_stream_setup_seconds=data_loader_seconds,
                    reconstruction_seconds=reconstruction_seconds,
                    warmup_seconds=warmup_seconds, warmup_calls=warmup_calls, pocket_ids=pocket_ids,
                    sampling_seconds=sample_seconds, generation_wall_seconds=time.perf_counter()-started,
                    actual_nfe_per_candidate=calls['graph_forward_calls']/saved,
                    batch_costs=batch_costs, torch_version=torch.__version__)
    def sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(block)
        return h.hexdigest()
    from scripts.stage1_runtime import runtime_metadata
    manifest['runtime'] = runtime_metadata()
    manifest['record_full_trajectory'] = not args.no_full_trajectory
    manifest['record_confidence_history'] = True
    manifest['requested_num_pockets'] = args.num_pockets
    manifest['num_pockets'] = len(pocket_ids)
    manifest['checkpoint_sha256'] = sha256(args.checkpoint)
    manifest['config_sha256'] = sha256(root / 'sample.yml')
    manifest['sampler_sha256'] = SAMPLER_SOURCE_SHA256_AT_IMPORT
    manifest['git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    if args.device.startswith('cuda'):
        manifest['gpu_name'] = torch.cuda.get_device_name(args.device)
    (root / 'run.json').write_text(json.dumps(manifest, indent=2))
    (root / 'DONE').touch()
    logger.info('Completed: %s', root)


if __name__ == '__main__':
    main()
