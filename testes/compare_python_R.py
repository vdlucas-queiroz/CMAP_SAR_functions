import sys, warnings
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else '..')
import numpy as np, pandas as pd
from scipy.cluster import hierarchy
from sar_classification import models, separability as sep, validation as va, fitting as ft
warnings.simplefilter('ignore')

cls, cls3 = ['F', 'VS', 'PA', 'SE'], ['F', 'PA', 'SE']
g, ga = pd.read_csv('gauss.csv'), pd.read_csv('gamma.csv')
rep = []
def check(name, py, r, rtol=1e-9):
    py, r = np.asarray(py, float), np.asarray(r, float)
    ok = np.allclose(py, r, rtol=rtol, atol=0, equal_nan=True)
    err = np.nanmax(np.abs(py - r) / np.maximum(np.abs(r), 1e-300))
    rep.append((name, 'OK' if ok else 'DIFERENTE', f'{err:.1e}'))
def same(name, a, b): rep.append((name, 'OK' if a == b else 'DIFERENTE', '-'))

# separabilidade
check('Bhattacharyya gaussiana', sep.bhattacharyya_matrix(g, cls).loc[cls, cls], pd.read_csv('R_B_gauss.csv', index_col=0).loc[cls, cls])
check('Bhattacharyya Gama', sep.bhattacharyya_matrix(ga, cls, 'gamma', 3).loc[cls, cls], pd.read_csv('R_B_gamma.csv', index_col=0).loc[cls, cls])
check('Jeffries-Matusita', sep.jeffries_matusita_matrix(g, cls).loc[cls, cls], pd.read_csv('R_JM.csv', index_col=0).loc[cls, cls])
check('Média JM', sep.mean_jm_distance(g, cls), pd.read_csv('R_meanJM.csv').iloc[0, 0])
d, r = sep.forward_selection_jm(g, cls, ['B1', 'B2', 'B3'])
same('Seleção de atributos (conjuntos)', list(d.Features) + list(r.Features), list(pd.read_csv('R_fs_dist.csv').Features) + list(pd.read_csv('R_fs_res.csv').Features))
check('Seleção de atributos (JM e ganho)', np.r_[d.JM_dist, r.Gain], np.r_[pd.read_csv('R_fs_dist.csv').JM_dist, pd.read_csv('R_fs_res.csv').Gain])
Z = sep.hierarchical_clustering(sep.bhattacharyya_matrix(g, cls))
check('Ward: alturas de fusão', np.sort(Z[:, 2]), np.sort(pd.read_csv('R_heights.csv').iloc[:, 0]))
same('extract_hierarchy', ['+'.join(sorted(v)) for v in sep.extract_hierarchy(Z, cls).values()], open('R_hier.txt').read().split())

# verossimilhanças
tr, te = pd.read_csv('gauss_train.csv'), pd.read_csv('gauss_test.csv')
lk = models.class_likelihood(te[['B1', 'B2', 'B3']], models.fit_model(tr, cls)).to_numpy()
lr = pd.read_csv('R_lik_gauss.csv').to_numpy(); m = lr > np.finfo(float).tiny
check('Gaussiana × R / (1/√2) (faixa normal)', lk[m] / lr[m], np.full(m.sum(), 1 / np.sqrt(2)))
same('Gaussiana: classe vencedora', (lk.argmax(1) == lr.argmax(1)).all(), True)
tr, te = pd.read_csv('gamma_train.csv'), pd.read_csv('gamma_test.csv')
check('Gama: verossimilhança', models.class_likelihood(te[['HH']], models.fit_model(tr, cls, 'gamma', 3)), pd.read_csv('R_lik_gamma.csv'), 1e-12)
tr, te = pd.read_csv('pair_train.csv'), pd.read_csv('pair_test.csv')
mp = models.fit_model(tr, cls3, 'intensity_joint_distribution', 4)
check('Par: parâmetros', pd.DataFrame(mp['params']).T[['ro2', 'mu1', 'mu2']], pd.read_csv('R_par_pair.csv'), 1e-12)
check('Par: verossimilhança', models.class_likelihood(te[['HH', 'HV']], mp), pd.read_csv('R_lik_pair.csv'), 1e-8)

# matriz de confusão e métricas
cm = pd.read_csv('cm.csv', index_col=0)
agg_r = pd.read_csv('R_agg.csv', index_col=0)
check('aggregate_classes', va.aggregate_classes(cm, ['F', 'VS']).loc[agg_r.index, agg_r.columns], agg_r)
mt = va.metrics_calculation(cm)
check('Métricas globais', [mt['overall_accuracy'], mt['f1_macro']], pd.read_csv('R_metrics_global.csv').iloc[0])
check('Métricas por classe', pd.concat([mt['producers_accuracy'], mt['users_accuracy'], mt['f1_score']], axis=1), pd.read_csv('R_metrics_class.csv', index_col=0))
for name, c, dist, enl in [('gauss', cls, 'gaussian', None), ('gamma', cls, 'gamma', 3), ('pair', cls3, 'intensity_joint_distribution', 4)]:
    out = va.evaluate_split(pd.read_csv(f'{name}_train.csv'), pd.read_csv(f'{name}_test.csv'), c, dist, enl)
    cm_r = pd.read_csv(f'R_cm_{name}.csv', index_col=0)
    check(f'Iteração completa ({dist})', out['confusion_matrix'].loc[cm_r.index, cm_r.columns], cm_r)

# ajustes
gf, gr = ft.gamma_fitting(ga, cls, verbose=False), pd.read_csv('R_gfit.csv')
check('Gama MV: α, β (otimizadores)', gf[['Alpha', 'Beta']], gr[['Alpha', 'Beta']], 1e-3)
check('Gama MV: p-valor', gf.p_value, gr.p_value, 1e-2)
check('Gama momentos: α, β, p', ft.gamma_fitting(ga, cls, method='mme', verbose=False)[['Alpha', 'Beta', 'p_value']], pd.read_csv('R_gfit_mme.csv')[['Alpha', 'Beta', 'p_value']], 1e-6)
check('Gama ENL fixo: β, p', ft.gamma_fitting_fixed_enl(ga, cls, 3)[['Beta', 'p_value']], pd.read_csv('R_gspec.csv')[['Rate', 'p_value']], 1e-6)
it, ir = ft.iterative_gamma_fitting(ga, cls, verbose=False), pd.read_csv('R_iter.csv')
check('ENL iterativo: candidatos', it['result_counting'].ENL, ir.ENL, 1e-3)
same('ENL iterativo: contagens', list(it['result_counting'].Count), list(ir.Count))
check('ENL iterativo: vencedor', it['best_enl'], pd.read_csv('R_bestenl.csv').iloc[0, 0], 1e-3)
check('Normal MV: μ, σ, p', ft.normal_fitting(g[['Class', 'B1']], cls, verbose=False)[['Mean', 'SD', 'p_value']], pd.read_csv('R_nfit.csv')[['Mean', 'SD', 'p_value']], 1e-6)

w = max(len(x[0]) for x in rep)
for n, s, e in rep: print(f'{n:<{w}}  {s:<10} {e}')
print(f'\n{sum(s == "OK" for _, s, _ in rep)} de {len(rep)} verificações OK')
