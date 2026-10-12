"""CPU regression: trace recording must not change sampling RNG/state/NFE."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from models.sample import sample_loop3
class Noise:
    num_steps=3
    def steps_loop(self,add_last=False):return range(3)
    def __call__(self,b,step):
        b['node_in']=b['node_type'].clone();b['halfedge_in']=b['halfedge_type'].clone()
        b['pos_in']=b['node_pos']+torch.randn_like(b['node_pos']);return b
    def outputs2batch(self,b,o):b['node_pos']=o['pred_pos'];return b
    def outputs2batch_ar(self,b,o,step,cfd):
        b['node_p2']=torch.arange(1 if step==0 else 0);return b
class Model:
    def __init__(self):self.calls=0
    def __call__(self,b):
        self.calls+=1
        return dict(pred_node=torch.nn.functional.one_hot(b['node_type'],3),pred_halfedge=torch.nn.functional.one_hot(b['halfedge_type'],3),pred_pos=b['pos_in']*.9,confidence_pos=b['pos_in'].sum(-1))
def run(record,ar):
    torch.manual_seed(2026);m=Model()
    b=dict(node_type=torch.tensor([1,2]),halfedge_type=torch.tensor([1]),node_pos=torch.zeros(2,3))
    b,o,t=sample_loop3(b,m,Noise(),is_ar=ar,off_tqdm=True,record_trajectory=record)
    return b,o,t,m.calls,torch.get_rng_state()
for ar in ['', 'ar']:
    a,b=run(True,ar),run(False,ar)
    for key in ['node_type','node_pos','halfedge_type']:assert torch.equal(a[0][key],b[0][key])
    for key in a[1]:assert torch.equal(a[1][key],b[1][key])
    assert a[2] is not None and b[2] is None and a[3]==b[3] and torch.equal(a[4],b[4])
print('PASS: trace on/off preserves states, predictions, confidence, NFE and RNG for standard and AR sampling')
