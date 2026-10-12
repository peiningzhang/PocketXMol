"""Recompute fresh ensembles; check seed/thread sensitivity without re-embedding output."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem
from posebusters import PoseBusters

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sdf',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
module=importlib.import_module('posebusters.modules.energy_ratio');original=module.new_conformation
cfg=copy.deepcopy(PoseBusters('mol').config);cfg['modules']=[m for m in cfg['modules'] if m['function'] in ['loading','energy_ratio']]
rows=[]
for path in a.sdf:
    mol=Chem.MolFromMolFile(path)
    if mol is None:continue
    values=[]
    for threads in [1,1,0]:
        module.new_conformation=lambda mol,n_confs=1,num_threads=0,energy_minimization=True: original(mol,n_confs,threads,energy_minimization)
        module.get_average_energy.cache_clear()
        result=PoseBusters(copy.deepcopy(cfg)).bust(mol,full_report=True).iloc[0]
        values.append(dict(threads=threads,ratio=float(result['energy_ratio']),energy_pass=bool(result['internal_energy'])))
    rows.append(dict(sdf=path,recomputations=values,fixed_thread_exact=values[0]['ratio']==values[1]['ratio'],thread_pass_sensitive=values[0]['energy_pass']!=values[2]['energy_pass']))
a.out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
assert all(r['fixed_thread_exact'] and not r['thread_pass_sensitive'] for r in rows)
