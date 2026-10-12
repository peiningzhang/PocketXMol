"""Concrete adjacent-round repair/damage examples with original SDF links."""
import json
from pathlib import Path
import pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
RDLogger.DisableLog('rdApp.*')
ROOT=Path('outputs_test_time_scaling/stage_1')
def main():
    f=pd.read_csv(ROOT/'tables/repair_damage_with_attribution_v2.csv');out=ROOT/'repair_examples';out.mkdir(exist_ok=True)
    text=['# 局部修复的具体例子','', '以下是保存的开发诊断子集中的相邻轮次，不是确认80口袋的统计结论。初始→当前的累计指标另保留在 CSV 中；本文使用前一轮→当前轮指标。2D 图片展示类型和键结构，3D 坐标变化以表中位移及原始 SDF 为准。',''];records=[]
    for field,title in [('P_phys_round_repair','物理修复'),('P_phys_round_damage','新增物理错误'),('valid_round_damage','新增化学或重构失效')]:
        g=f[f[field].astype(str).str.lower().eq('true')]
        if not len(g):text+=['## '+title,'','保存子集中没有该类相邻轮次事件。',''];continue
        if field!='valid_round_damage':
            valid=g[g.valid_previous.astype(str).str.lower().eq('true')&g.valid_final.astype(str).str.lower().eq('true')]
            if len(valid):g=valid
        row=g.sort_values(['run','data_id','round']).iloc[0];selected=json.loads(row.selected_atom_indices_local)
        before=Path(row.previous_sdf);after=Path(row.final_sdf);stem=field
        text+=['## '+title,'',f"运行 `{row.run}`，口袋 `{row.data_id}`，seed={row.seed}，相邻轮次 {int(row['round'])-1}→{int(row['round'])}。选中图内原子编号：{selected}。",'',f'[前一轮 SDF]({before}) · [当前轮 SDF]({after})','', '| 指标 | 前一轮 | 当前轮 |','|---|---:|---:|']
        for metric in ['P_phys','P_joint','valid','complete','geometry_pass','energy_pass','pocket_no_clash','n_atoms','energy_ratio','vina_score']:
            text.append(f"| {metric} | {row[metric+'_previous']} | {row[metric+'_final']} |")
        text+=['',f"原始模型图的原子类型变化 {int(row.raw_atom_type_changes)} 处、半边类型变化 {int(row.raw_edge_type_changes)} 处；全部原子坐标 RMS 位移 {row.raw_position_rms_displacement:.3f} Å，选中原子的 SDF RMS 位移 {row.sdf_selected_position_rms:.3f} Å。未选坐标最大位移 {row.sdf_unselected_position_max:.3f} Å；原始固定上下文与口袋断言均为 {row.raw_fixed_context_exact}/{row.raw_pocket_exact}。",'',f"SDF 原子类型变化编号：{row.sdf_atom_type_changed_indices}。未选内部连接关系增删：{int(row.sdf_unselected_connectivity_changes)}。重构后的键级/芳香性变化与原始模型半边更新分别记录，不能混同。",'']
        mols=[Chem.MolFromMolFile(str(p),sanitize=True,removeHs=False) for p in [before,after]]
        canonical=[Chem.MolToSmiles(m,isomericSmiles=True) if m is not None else None for m in mols]
        same_identity=canonical[0]==canonical[1] if all(x is not None for x in canonical) else None
        text+=['规范含立体信息 SMILES 是否一致：'+str(same_identity)+'。','']
        if same_identity is False:text+=['规范 SMILES 改变；前后能量比各自对应各自分子的50构象 ensemble。','']
        drawing=None
        if all(m is not None for m in mols):
            for mol in mols:rdDepictor.Compute2DCoords(mol)
            drawer=rdMolDraw2D.MolDraw2DSVG(900,350,450,350);drawer.drawOptions().addAtomIndices=True
            drawer.DrawMolecules(mols,legends=['Previous round','Current round'],highlightAtoms=[selected,selected]);drawer.FinishDrawing();drawing=out/(stem+'.svg');drawing.write_text(drawer.GetDrawingText());
            raster=rdMolDraw2D.MolDraw2DCairo(900,350,450,350);raster.drawOptions().addAtomIndices=True;raster.DrawMolecules(mols,legends=['Previous round','Current round'],highlightAtoms=[selected,selected]);raster.FinishDrawing();(out/(stem+'.png')).write_bytes(raster.GetDrawingText())
            text+=['!['+title+']('+str(drawing.resolve())+')','']
        else:text+=['化学清洗失败；原始 SDF 已保留供复查。','']
        records.append(dict(event=field,run=row.run,data_id=row.data_id,seed=int(row.seed),round=int(row['round']),before_sdf=str(before),after_sdf=str(after),selected_atom_indices_local=selected,canonical_smiles_before=canonical[0],canonical_smiles_after=canonical[1],chemical_identity_unchanged=same_identity,drawing=str(drawing) if drawing else None))
    (ROOT/'repair_examples.md').write_text('\n'.join(text));(out/'manifest.json').write_text(json.dumps(records,indent=2));print('Concrete diagnostic examples:',len(records))
if __name__=='__main__':main()
