"""Fixed-context 25% partial regeneration using teacher maskfill noise path.

Unlike native radius/threshold refinement, select exactly ceil(0.25*n) atoms
per graph. Update their types and positions plus all incident bonds. Keep
unselected atom types/coordinates and internal context bonds exactly fixed.
"""
import math
import torch
from models.sample import sample_loop3


def partition(batch, outputs, selection, fraction=.25):
    selected=torch.zeros_like(batch.node_type,dtype=torch.bool)
    for graph in range(batch.num_graphs):
        idx=torch.where(batch.node_type_batch==graph)[0]
        count=max(1,math.ceil(len(idx)*fraction))
        if selection=='random':
            chosen=idx[torch.randperm(len(idx),device=idx.device)[:count]]
        elif selection=='confidence':
            confidence=outputs['confidence_pos'].flatten()[idx]
            chosen=idx[torch.argsort(confidence,stable=True)[:count]]
        else:raise ValueError(selection)
        selected[chosen]=True
    edge_selected=selected[batch.halfedge_index].any(dim=0)
    node_p2=torch.where(selected)[0];node_p1=torch.where(~selected)[0]
    endpoints=selected[batch.halfedge_index]
    batch.update({'node_p1':node_p1,'node_p2':node_p2,
        'halfedge_p1':torch.where(~edge_selected)[0],
        'halfedge_p2':torch.where(endpoints.all(dim=0))[0],
        'halfedge_p1p2':torch.where(endpoints.any(dim=0)&~endpoints.all(dim=0))[0],
        'fixed_node':(~selected).long(),'fixed_pos':(~selected).long(),
        'fixed_halfedge':(~edge_selected).long(),
        'fixed_halfdist':(~edge_selected).long()})
    assert torch.equal(batch.fixed_halfdist, batch.fixed_halfedge)
    return selected,edge_selected


def partial_loop(batch,model,noiser,device=None,is_ar='',off_tqdm=False,*,selection,rounds,noise_start=1.0,diagnostics=None,states=None,record_trajectory=True):
    batch,outputs,_=sample_loop3(batch,model,noiser,device,is_ar='',off_tqdm=off_tqdm,record_trajectory=record_trajectory)
    confidence=[outputs['confidence_pos_traj']]
    if states is not None:
        states.append((0,batch.clone(),{k:v.clone() for k,v in outputs.items() if isinstance(v,torch.Tensor)}))
    for round_index in range(rounds):
        selected,edges=partition(batch,outputs,selection)
        snapshot={k:batch[k].clone() for k in ['node_type','node_pos','halfedge_type','pocket_pos','halfedge_index']}
        noiser.init_step=noise_start
        round_confidence=[]
        for step in noiser.steps_loop(add_last=False):
            with torch.no_grad():
                batch=noiser(batch,step)
                # Assert fixed inputs before every model call, not just final output.
                assert torch.equal(batch.node_in[~selected],snapshot['node_type'][~selected])
                assert torch.equal(batch.pos_in[~selected],snapshot['node_pos'][~selected])
                assert torch.equal(batch.halfedge_in[~edges],snapshot['halfedge_type'][~edges])
                outputs=model(batch)
                batch.update({'step':step})
                batch=noiser.outputs2batch(batch,outputs)
                assert torch.equal(batch.node_type[~selected],snapshot['node_type'][~selected])
                assert torch.equal(batch.node_pos[~selected],snapshot['node_pos'][~selected])
                assert torch.equal(batch.halfedge_type[~edges],snapshot['halfedge_type'][~edges])
                assert torch.equal(batch.pocket_pos,snapshot['pocket_pos'])
                assert torch.equal(batch.halfedge_index,snapshot['halfedge_index'])
                round_confidence.append(outputs['confidence_pos'].detach().flatten())
        confidence.append(torch.stack(round_confidence,dim=-1))
        if states is not None:
            stage_outputs={k:v.clone() for k,v in outputs.items() if isinstance(v,torch.Tensor)}
            stage_outputs['confidence_pos_traj']=confidence[-1].clone()
            states.append((round_index+1,batch.clone(),stage_outputs))
        if diagnostics is not None:
            for graph in range(batch.num_graphs):
                nodes=batch.node_type_batch==graph; edge_mask=batch.halfedge_type_batch==graph
                diagnostics.append(dict(data_id=batch.data_id[graph],round=round_index+1,
                    selected_atom_indices=torch.where(selected&nodes)[0].cpu().tolist(),
                    selected_atom_index_space='batch_global',
                    selected_atom_indices_local=torch.where(selected[nodes])[0].cpu().tolist(),
                    atom_count=int(nodes.sum()),selected_count=int((selected&nodes).sum()),
                    updated_edge_count=int((edges&edge_mask).sum()),
                    atom_type_changes=int(((batch.node_type!=snapshot['node_type'])&nodes).sum()),
                    edge_type_changes=int(((batch.halfedge_type!=snapshot['halfedge_type'])&edge_mask).sum()),
                    position_rms_displacement=float((batch.node_pos[nodes]-snapshot['node_pos'][nodes]).square().sum(-1).mean().sqrt()),
                    fixed_context_exact=True,pocket_exact=True,
                    teacher_fixed_distance_conditioning=True))
    outputs['confidence_pos_traj']=torch.cat(confidence,dim=-1)
    return batch,outputs,None
