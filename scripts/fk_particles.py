"""Native no-AR FK-style particle adaptation with clean-prediction potentials.

Resampling copies whole graphs (including pocket conditioning), confidence
history and parent reward history. No energy or Vina enters the potential.
"""
import time
import torch
from torch_geometric.data import Batch


def clean_reward(batch, outputs):
    rewards=[]
    pos=outputs['pred_pos']
    edges=outputs['pred_halfedge'].argmax(-1)
    for graph in range(batch.num_graphs):
        nodes=batch.node_type_batch==graph
        local=pos[nodes]
        pocket=batch.pocket_pos[batch.pocket_pos_batch==graph]
        edge_mask=batch.halfedge_type_batch==graph
        edge_index=batch.halfedge_index[:,edge_mask]
        distances=(pos[edge_index[0]]-pos[edge_index[1]]).norm(dim=-1)
        bonded=edges[edge_mask]>0
        if not torch.isfinite(local).all() or not torch.isfinite(distances).all():
            rewards.append(pos.new_tensor(float('nan')));continue
        # Broad bond-distance and steric soft penalties, normalized by atom count.
        bond_penalty=((.8-distances).clamp(min=0)+(distances-2.0).clamp(min=0))[bonded].sum()
        internal_penalty=(.8-distances[~bonded]).clamp(min=0).sum()
        pocket_penalty=(1.4-torch.cdist(local,pocket)).clamp(min=0).sum() if len(pocket) else pos.new_tensor(0.)
        rewards.append(-(bond_penalty+internal_penalty+pocket_penalty)/max(len(local),1))
    return torch.stack(rewards)


def batched_forward(batch, model, follow_batch, max_batch=4):
    if batch.num_graphs<=max_batch:return model(batch)
    graphs=batch.to_data_list(); results=[]
    for key, assignment in [('node_in',batch.node_type_batch),('pos_in',batch.node_type_batch),
                            ('halfedge_in',batch.halfedge_type_batch),('log_sigma',batch.node_type_batch)]:
        if key in batch:
            for graph, data in enumerate(graphs):
                data[key]=batch[key][assignment==graph].clone()
    for start in range(0,len(graphs),max_batch):
        piece=Batch.from_data_list(graphs[start:start+max_batch],follow_batch=follow_batch).to(batch.node_pos.device)
        piece['step']=batch['step']
        results.append(model(piece))
    return {key:torch.cat([r[key] for r in results],dim=0) for key in results[0]}


def fk_loop(batch, model, noiser, *, interval, strength, follow_batch, diagnostics, max_batch=4):
    particle_count=batch.num_graphs
    assert len(set(batch.data_id))==1,'Cannot mix pockets in particle population'
    ancestors=torch.arange(particle_count,device=batch.node_pos.device)
    previous=torch.zeros(particle_count,device=batch.node_pos.device)
    histories=[[] for _ in range(particle_count)]
    verifier_seconds=0.0
    for call,step in enumerate(noiser.steps_loop(add_last=False),1):
        with torch.no_grad():
            batch=noiser(batch,step);batch['step']=step
            outputs=batched_forward(batch,model,follow_batch,max_batch)
            batch=noiser.outputs2batch(batch,outputs)
            chunks=torch.split(outputs['confidence_pos'].detach().flatten(),batch.node_type_batch.bincount().tolist())
            for hist,chunk in zip(histories,chunks):hist.append(chunk)
            if call%interval or call==noiser.num_steps:continue
            tick=time.perf_counter(); reward=clean_reward(batch,outputs)
            finite=torch.isfinite(reward)
            uniform=not bool(finite.any())
            if uniform:
                weights=torch.full_like(reward,1/particle_count)
                safe_reward=previous.clone()
            else:
                safe_reward=torch.where(finite,reward,previous)
                log_weights=strength*(safe_reward-previous)
                log_weights=torch.where(finite,log_weights,torch.full_like(log_weights,-torch.inf))
                weights=torch.softmax(log_weights-log_weights[finite].max(),dim=0)
            parents=torch.multinomial(weights,particle_count,replacement=True)
            graphs=batch.to_data_list()
            copied=[graphs[int(i)].clone() for i in parents]
            # Assert every tensor and every metadata field copied from exact parent graph.
            for child,parent in zip(copied,parents):
                reference=graphs[int(parent)]
                keys = reference.keys() if callable(reference.keys) else reference.keys
                for key in keys:
                    if isinstance(reference[key],torch.Tensor):
                        assert torch.equal(child[key],reference[key]),f'Incomplete parent copy: {key}'
                    else:assert child[key]==reference[key],f'Metadata mismatch: {key}'
            old_histories=histories
            histories=[list(old_histories[int(i)]) for i in parents]
            ancestors=ancestors[parents]
            previous=safe_reward[parents].clone()
            batch=Batch.from_data_list(copied,follow_batch=follow_batch).to(batch.node_pos.device)
            assert len(set(batch.data_id))==1
            elapsed=time.perf_counter()-tick;verifier_seconds+=elapsed
            diagnostics.append(dict(data_id=batch.data_id[0],call=call,reward=reward.cpu().tolist(),
                weights=weights.cpu().tolist(),ESS=float(1/weights.square().sum()),parents=parents.cpu().tolist(),
                original_ancestor_count=int(ancestors.unique().numel()),unique_parents=int(parents.unique().numel()),
                uniform_fallback=uniform,full_parent_copy_exact=True,seconds=elapsed))
    outputs['confidence_pos_traj']=torch.cat([torch.stack(h,dim=-1) for h in histories],dim=0)
    return batch,outputs,verifier_seconds
