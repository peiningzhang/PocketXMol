"""Regression: aromatic repair must propagate its existing timeout."""
from unittest.mock import patch
from rdkit import Chem
from utils.misc import TimeoutException
from utils.reconstruct import fix_aromatic

mol=Chem.MolFromSmiles('c1ccccc1')
for side_effect in [TimeoutException('test'),[ValueError('not sanitizable'),TimeoutException('test')]]:
    with patch('utils.reconstruct.Chem.SanitizeMol',side_effect=side_effect):
        try:fix_aromatic(mol)
        except TimeoutException:pass
        else:raise AssertionError('Aromatic repair swallowed timeout')
print('PASS: both aromatic repair branches propagate TimeoutException')
