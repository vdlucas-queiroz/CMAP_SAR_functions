import sys, glob, warnings; sys.path.insert(0, '..'); warnings.simplefilter('ignore')
import numpy as np, pandas as pd, rasterio
from sar_classification import sampling as smpl, transitions as trn, trajectory_assessment as ta, masking
rep = []
def ok(name, cond, det='-'): rep.append((name, 'OK' if cond else 'DIFERENTE', det))
cn = ['A', 'B', 'C', 'D', 'E']
key = lambda df, cls: set(zip(np.round(df['x'], 3), np.round(df['y'], 3), df[cls]))

pts = smpl.raster_samples_to_points('samples.tif', cn)
r = pd.read_csv('R_out/points/p_points.csv')
ok('raster_to_points: mesmos pontos e classes', key(r, 'class_name') == set(zip(np.round(pts.geometry.x, 3), np.round(pts.geometry.y, 3), pts.class_name)), f'{len(pts)} pontos')

lags = pd.DataFrame({'h': [3, 2, 1, 4, 2], 'v': [2, 3, 1, 2, 5]}, index=cn)
tr, va = smpl.lag_grid_split('samples.tif', cn, lags)
rt, rv = pd.read_csv('R_out/train/lag_train.csv'), pd.read_csv('R_out/valid/lag_valid.csv')
pk = lambda g: set(zip(np.round(g.geometry.x, 3), np.round(g.geometry.y, 3), g.class_name))
ok('Amostragem por lags: treino idêntico', key(rt, 'class_name') == pk(tr), f'{len(tr)} pontos')
ok('Amostragem por lags: validação idêntica', key(rv, 'class_name') == pk(va), f'{len(va)} pontos')

for tag, bal in [('rand', None), ('bal2', 2)]:
    rtr = pd.read_csv(glob.glob(f'R_out/{tag}/train/*.csv')[0]); rva = pd.read_csv(glob.glob(f'R_out/{tag}/valid/*.csv')[0])
    t2, v2 = smpl.random_split_points(pts, 0.7, balance=bal, random_state=1)
    cr, cp = rtr.class_name.value_counts().sort_index(), t2.class_name.value_counts().sort_index()
    ok(f'Divisão aleatória (balance={bal}): nº de treino por classe', cr.equals(cp), str(cp.to_dict()))
    ok(f'Divisão aleatória (balance={bal}): treino ∪ validação = todos os pixels', len(rtr) + len(rva) == len(t2) + len(v2) == len(pts))

mats = [trn.read_transition_matrix(f'T{t}.txt', has_labels=False, verbose=False) for t in (1, 2, 3)]
imp = ta.impossible_trajectories([f'ML_{t}.tif' for t in (1, 2, 3, 4)], mats, output_path='py_imp.tif')
r_pct = 17.53909
ok('Trajetórias impossíveis: percentual', abs(imp['percentage'] - r_pct) < 1e-4, f"Python {imp['percentage']:.5f} × R {r_pct}")
ri = pd.read_csv('R_out/impossible.csv')['v'].to_numpy()
with rasterio.open('py_imp.tif') as s: pi = s.read(1).astype(float).ravel(); pi[pi == 255] = np.nan
ok('Trajetórias impossíveis: raster pixel a pixel', np.array_equal(np.nan_to_num(ri, nan=-1), np.nan_to_num(pi, nan=-1)))

cmp = ta.compare_classifications([f'ML_{t}.tif' for t in (1, 2, 3, 4)], [f'CMAP_{t}.tif' for t in (1, 2, 3, 4)], 'cloud.tif', verbose=False)
r_dates, r_total = [10.91549, 10.97245, 11.41649, 11.49425], 38.37675
ok('MaxVer × CMAP: % por data', np.allclose(cmp['per_date'].percentage, r_dates, atol=1e-5), str(np.round(cmp['per_date'].percentage.to_numpy(), 5)))
ok('MaxVer × CMAP: % total', abs(cmp['total_percentage'] - r_total) < 1e-5, f"{cmp['total_percentage']:.5f} × {r_total}")
ra = pd.read_csv('R_out/accumulated.csv')['v'].to_numpy(); pa = cmp['accumulated'].astype(float).ravel(); pa[pa == -9999] = np.nan
ok('MaxVer × CMAP: raster acumulado', np.array_equal(np.nan_to_num(ra, nan=-9), np.nan_to_num(pa, nan=-9)))

with rasterio.open('img.tif') as s, rasterio.open('cloud.tif') as m:
    py = masking.mask_array(s.read(), m.read(1), -1)
rm = pd.read_csv('R_out/masked.csv').to_numpy()
ok('Máscara de nuvens', np.allclose(py.reshape(2, -1).T, rm, rtol=1e-6))

w = max(len(x[0]) for x in rep)
for n, s, d in rep: print(f'{n:<{w}}  {s:<10} {d}')
print(f"\n{sum(s == 'OK' for _, s, _ in rep)} de {len(rep)} verificações OK")
