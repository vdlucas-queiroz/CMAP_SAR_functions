"""CMAP (máscara por data, saída multibanda) × CMAP_classifier do R (CMAP_C4_C5.R); MaxVer × argmax."""
import sys, subprocess, itertools; sys.path.insert(0, '..')
import numpy as np, pandas as pd, rasterio
from rasterio.transform import from_origin
from sar_classification import cmap
rng = np.random.default_rng(8); H, W = 30, 40; nb = [3, 2, 3]
for t, k in enumerate(nb):
    with rasterio.open(f'lik{t+1}.tif', 'w', driver='GTiff', height=H, width=W, count=k, dtype='float32', crs='EPSG:31983', transform=from_origin(0, 0, 30, 30)) as d:
        d.write(rng.gamma(2, 1, (k, H, W)).astype('float32'))
for t in (1, 3):
    with rasterio.open(f'mask{t}.tif', 'w', driver='GTiff', height=H, width=W, count=1, dtype='uint8', crs='EPSG:31983', transform=from_origin(0, 0, 30, 30)) as d:
        d.write((rng.random((H, W)) < 0.2).astype('uint8'), 1)
rows = [{'S1': a, 'S2': b, 'S3': c, 'final_weights': (0 if (w := rng.random()) < 0.4 else w)} for a, b, c in itertools.product(range(1, 4), range(1, 3), range(1, 4))]
pd.DataFrame(rows).to_csv('pesos.txt', sep=';', index=False)
subprocess.run(['Rscript', 'run_r_cmap_classifier.R'], check=True, capture_output=True)
tw = pd.read_csv('pesos.txt', sep=';')
wdf = pd.DataFrame({'Stage_1': tw.S1.astype(str), 'Stage_2': tw.S2.astype(str), 'Stage_3': tw.S3.astype(str), 'final_weight': tw.final_weights})
cfg = [['1', '2', '3'], ['1', '2'], ['1', '2', '3']]; codes = {'1': 1, '2': 2, '3': 3}
o = cmap.cmap_classifier(['lik1.tif', 'lik2.tif', 'lik3.tif'], wdf, cfg, 'py_out', class_codes=codes, masks=['mask1.tif', None, 'mask3.tif'],
                         mask_mode='output', output_layout='stack', verbose=False)
py = rasterio.open(o['classified']).read()
print('CMAP Python == CMAP_classifier do R:', all(np.array_equal(py[i], rasterio.open(f'R_out/CMAP_{i+1}_teste.tif').read(1).astype(int)) for i in range(3)))
mv = rasterio.open(cmap.maxver_from_likelihoods(['lik1.tif', 'lik2.tif', 'lik3.tif'], 'py_out/maxver.tif', codes, cfg)).read()
print('maxver_from_likelihoods == argmax:', all(np.array_equal(mv[i], np.argmax(rasterio.open(f'lik{t}.tif').read(), 0) + 1) for i, t in enumerate((1, 2, 3))))
