"""Compara autocorrelation.py com as funções ORIGINAIS de correlation_2.ipynb (copiadas sem alteração)."""
import sys, json, warnings, io, contextlib
sys.path.insert(0, '..'); warnings.simplefilter('ignore')
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon
from scipy.ndimage import gaussian_filter

# --- código original (células 0 e 3–6 do notebook)
nb = json.load(open('originais_correlation_2.ipynb'))
for i in (0, 3, 4, 5, 6):
    exec(''.join(nb['cells'][i]['source']))

# --- dados fictícios: 2 bandas de intensidade com correlação espacial de alcances diferentes por região
rng = np.random.default_rng(3)
H, W = 300, 300
bands = []
for b in range(2):
    field = np.zeros((H, W))
    for (r0, c0), sig in zip([(0, 0), (0, 150), (150, 0), (150, 150)], [0.8, 1.5, 2.5, 4.0]):
        f = gaussian_filter(rng.normal(size=(150, 150)), sig)
        field[r0:r0 + 150, c0:c0 + 150] = f / f.std()
    bands.append(np.exp(0.5 * field + b * 0.1) * 0.05)
img = np.stack(bands).astype('float32')
img[:, 10:14, 10:14] = -9999                                   # nodata dentro de um polígono
tr = from_origin(500000, 9000000, 12.5, 12.5)
with rasterio.open('img.tif', 'w', driver='GTiff', height=H, width=W, count=2, dtype='float32',
                   crs='EPSG:31983', transform=tr, nodata=-9999) as dst: dst.write(img)

polys, cls = [], []
for k in range(40):
    r, c = rng.integers(2, 120, 2) + (150 if k % 4 >= 2 else 0, 150 if k % 2 else 0)
    h, w = rng.integers(12, 28, 2)
    x0, y0 = tr * (c + 0.3, r + 0.6); x1, y1 = tr * (c + w + 0.2, r + h + 0.1)
    polys.append(Polygon([(x0, y0), (x1, y0), (x1 - 30, y1), (x0 + 20, y1)]))       # trapézios (bordas parciais)
    cls.append(['A', 'B', 'C', 'D'][k % 4])
polys[0] = Polygon([(tr * (8, 8)), (tr * (20, 8)), (tr * (20, 20)), (tr * (8, 20))]); cls[0] = 'A'   # contém nodata
gdf = gpd.GeoDataFrame({'Classe': cls}, geometry=polys, crs='EPSG:31983')

# --- execução original (laço da célula 12, sem prints)
orig = []
with rasterio.open('img.tif') as src, contextlib.redirect_stdout(io.StringIO()):
    for banda in (1, 2):
        for idx, row in gdf.iterrows():
            masked_band, _ = rio_mask(src, [row['geometry']], crop=False, indexes=banda, invert=False, filled=True, nodata=-1)
            masked_band[masked_band == src.nodata] = -1
            arr = extrair_dados_raster(row['geometry'], masked_band, src.transform)
            lh, lv = autocorrelation(arr, '', target_correlation=0.3, plot=False)
            orig.append({'polygon': idx, 'band': banda, 'lag_h': np.nan if lh is None else lh, 'lag_v': np.nan if lv is None else lv})
orig = pd.DataFrame(orig)

from sar_classification import autocorrelation as ac
new = ac.sample_lags('img.tif', gdf, 'Classe', target_correlation=0.3, verbose=False)
m = orig.merge(new, on=['polygon', 'band'], suffixes=('_orig', '_new'))
eq = ((m.lag_h_orig == m.lag_h_new) | (m.lag_h_orig.isna() & m.lag_h_new.isna())) & \
     ((m.lag_v_orig == m.lag_v_new) | (m.lag_v_orig.isna() & m.lag_v_new.isna()))
print(f'Lags por polígono e banda idênticos ao original: {int(eq.sum())} de {len(m)}')
print('Faixa de lags encontrados: h', sorted(new.lag_h.dropna().unique().astype(int)), '| v', sorted(new.lag_v.dropna().unique().astype(int)))

# resumo: original (células 12 e 15) x novo
o = orig.assign(Classe=gdf.loc[orig.polygon, 'Classe'].values)
orig_max = {}
for banda in (1, 2):
    for c, g in o[o.band == banda].groupby('Classe'):
        mh = find_max_lag_without_outliers(list(g.lag_h.dropna().astype(int)), IQR=True)
        mv = find_max_lag_without_outliers(list(g.lag_v.dropna().astype(int)), IQR=True)
        d = orig_max.setdefault(c, {'h': 0, 'v': 0}); d['h'] = max(d['h'], mh); d['v'] = max(d['v'], mv)
orig_tab = pd.DataFrame(orig_max).T.sort_index()
new_tab = ac.lags_for_sampling(ac.summarize_lags(new))
print('\nLags finais por classe — original:'); print(orig_tab.T.to_string())
print('Lags finais por classe — novo:'); print(new_tab.T.to_string())
print('Idênticos:', orig_tab.astype(int).equals(new_tab[['h', 'v']].astype(int)))
