import numpy as np, rasterio
from rasterio.transform import from_origin
rng = np.random.default_rng(11)
H, W = 60, 80
tr = from_origin(400000, 8000000, 30, 30)
def write(path, a, dtype, nodata=None):
    with rasterio.open(path, 'w', driver='GTiff', height=H, width=W, count=a.shape[0] if a.ndim == 3 else 1,
                       dtype=dtype, crs='EPSG:31983', transform=tr, nodata=nodata) as d:
        d.write(a if a.ndim == 3 else a[None])
# raster de amostras: blocos de 5 classes + fundo 0
s = np.zeros((H, W), np.int16)
for k in range(1, 6):
    for _ in range(4):
        r, c = rng.integers(0, H - 8), rng.integers(0, W - 10)
        s[r:r + rng.integers(4, 9), c:c + rng.integers(5, 11)] = k
write('samples.tif', s, 'int16')
# série classificada (4 datas, classes 1..4) com mudanças e alguns NA
base = rng.integers(1, 5, (H, W))
for t in range(4):
    a = base.copy(); flip = rng.random((H, W)) < 0.25; a[flip] = rng.integers(1, 5, flip.sum())
    a = a.astype('float32'); a[rng.random((H, W)) < 0.02] = np.nan
    write(f'ML_{t + 1}.tif', a, 'float32', np.nan)
    b = a.copy(); flip = rng.random((H, W)) < 0.15; b[flip & ~np.isnan(a)] = rng.integers(1, 5, (flip & ~np.isnan(a)).sum())
    write(f'CMAP_{t + 1}.tif', b, 'float32', np.nan)
cloud = (rng.random((H, W)) < 0.1).astype('uint8'); write('cloud.tif', cloud, 'uint8')
img = rng.gamma(3, 0.03, (2, H, W)).astype('float32'); write('img.tif', img, 'float32')
# matrizes de transição numéricas, sem rótulos (formato do R), com zeros
for t in range(3):
    M = rng.random((4, 4)); M[rng.random((4, 4)) < 0.35] = 0; np.fill_diagonal(M, 1); M /= M.sum(1, keepdims=True)
    np.savetxt(f'T{t + 1}.txt', M, delimiter=';', fmt='%.6f')
print('ok')
