import sys, tempfile, warnings, io, contextlib; sys.path.insert(0, '..'); warnings.simplefilter('ignore')
import numpy as np, pandas as pd, rasterio
from pathlib import Path
from rasterio.transform import from_origin
# versão validada do notebook CMAP_FUNCTIONS (v1)
for f in ['c_imports.py', 'c_trans.py', 'c_cmap.py']: exec(open(f'referencia_cmap_v1/{f}').read())
from sar_classification import transitions as trn, cmap, models, raster_io as rio, trajectory_assessment as ta

rng = np.random.default_rng(5); d = Path(tempfile.mkdtemp()); H, W = 40, 50
cfg = [['F', 'PA', 'SE', 'VS'], ['F', 'PA', 'SE'], ['F', 'PA', 'SE', 'VS']]
tifs = []
for t, c in enumerate(cfg):
    a = rng.gamma(2, 1, (len(c), H, W)).astype('float32'); a[:, 0, :3] = -9999
    p = d / f'lik{t}.tif'; tifs.append(p)
    with rasterio.open(p, 'w', driver='GTiff', height=H, width=W, count=len(c), dtype='float32', crs='EPSG:31983',
                       transform=from_origin(0, 0, 30, 30), nodata=-9999) as dst:
        dst.write(a); [dst.set_band_description(i + 1, n) for i, n in enumerate(c)]
for t, (r, c) in enumerate([(cfg[0], cfg[1]), (cfg[1], cfg[2])]):
    m = rng.random((len(r), len(c))); m[m < 0.3] = 0; m /= m.sum(1, keepdims=True)
    pd.DataFrame(m, index=r, columns=c).to_csv(d / f'm{t}.txt', sep=';')
mats = [d / 'm0.txt', d / 'm1.txt']
with contextlib.redirect_stdout(io.StringIO()):
    w_old = create_transition_matrix_table(mats, d / 'old.csv')
    o_old = cmap_classifier(tifs, w_old, cfg, d / 'old')
    w_new = trn.create_transition_matrix_table(mats, verbose=False)
    o_new = cmap.cmap_classifier(tifs, w_new, None, d / 'new', verbose=False)      # bands_config lido das bandas
rd = lambda p: rasterio.open(p).read(1)
print('Tabela de P(s) idêntica à do notebook:', np.allclose(w_old.final_weight, w_new.final_weight))
print('Trajectory_IDs idênticos:', np.array_equal(rd(o_old['trajectory_ids']), rd(o_new['trajectory_ids'])))
print('Mapas por data idênticos:', all(np.array_equal(rd(a), rd(b)) for a, b in zip(o_old['classified'], o_new['classified'])))

# entrada em log-verossimilhança -> mesmo resultado
logt = []
for p in tifs:
    with rasterio.open(p) as s:
        a = s.read().astype('float64'); a[a == -9999] = np.nan; prof = s.profile; desc = s.descriptions
    prof.update(nodata=np.nan); q = Path(str(p).replace('.tif', '_log.tif')); logt.append(q)
    with rasterio.open(q, 'w', **prof) as dst:
        dst.write(np.log(a).astype('float32')); [dst.set_band_description(i + 1, n) for i, n in enumerate(desc)]
o_log = cmap.cmap_classifier(logt, w_new, None, d / 'log', input_is_log=True, verbose=False)
print('Entrada em log -> mesmos Trajectory_IDs:', np.array_equal(rd(o_new['trajectory_ids']), rd(o_log['trajectory_ids'])))

# codificação comum MaxVer x CMAP
codes = {'F': 10, 'PA': 20, 'SE': 30, 'VS': 40}
o_c = cmap.cmap_classifier(tifs, w_new, None, d / 'codes', class_codes=codes, verbose=False)
print('class_codes aplicado:', sorted(np.unique(rd(o_c['classified'][0]))), '(0 = nodata)')
# CMAP nunca produz trajetória impossível
leg = {v: k for k, v in codes.items()}
mt = [trn.read_transition_matrix(m, verbose=False) for m in mats]
imp = ta.impossible_trajectories(o_c['classified'], mt, legends=leg)
print(f"Trajetórias impossíveis no resultado do CMAP: {imp['percentage']:.2f}% (esperado 0)")
