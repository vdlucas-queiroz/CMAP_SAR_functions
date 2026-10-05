"""Verificação de coerência de todas as distribuições do catálogo (não depende do R)."""
import sys, warnings
sys.path.insert(0, '..')
warnings.simplefilter('ignore')
import numpy as np
from sar_classification import diagnostics as dg

CASES = [('gaussian', {'mean': np.array([0.2]), 'covariance_matrix': np.array([[9e-4]])}, None),
         ('gaussian', {'mean': np.array([0.2, 0.05]), 'covariance_matrix': np.array([[9e-4, 2e-4], [2e-4, 4e-4]])}, None),
         ('gamma', {'alpha': 3.0, 'beta': 60.0}, 3.0),
         ('intensity_joint_distribution', {'mu1': 0.08, 'mu2': 0.02, 'ro2': 0.25}, 4)]
for name, params, enl in CASES:
    print(f'--- {name}')
    print(dg.check_distribution(name, params, enl).to_string(index=False), '\n')
