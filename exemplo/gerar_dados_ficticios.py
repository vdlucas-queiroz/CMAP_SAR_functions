"""
Gera um projeto fictício com a mesma estrutura do projeto real, para testar os 3 notebooks:
- SAR 2008 (bandas HV, HH) e 2010 (HH, HV), 10 classes, speckle Gama (L = 4) e textura espacial;
- raster de amostras (códigos 1..10) e polígonos de amostra por data;
- verossimilhanças ópticas 2007 e 2009 (legenda de 6 classes, SEM nomes de banda, como as do R),
  máscara de nuvens de 2007 e polígonos de validação com campo DN;
- matrizes de transição (com rótulos) óptico -> SAR -> óptico;
- config_exemplo.json.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter
from shapely.geometry import box

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / 'dados').resolve()
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2010)
H, W, RES = 100, 150, 12.5
TR = from_origin(600000, 8900000, RES, RES); CRS = 'EPSG:31981'
CLASSES = ['AP', 'AC', 'FD', 'FP', 'PL', 'PS', 'SE', 'VS1', 'VS2', 'VS3']
OPT = {'AP': 'AG', 'AC': 'AG', 'FD': 'F', 'FP': 'F', 'PL': 'PA', 'PS': 'PA', 'SE': 'SE', 'VS1': 'VS1', 'VS2': 'VS3', 'VS3': 'VS3'}
OPT_CLASSES = ['AG', 'VS3', 'SE', 'F', 'VS1', 'PA']                 # ordem das bandas ópticas (como classes_2005 no R)
# médias de intensidade (HH, HV) por classe
MU = {'AP': (.045, .006), 'AC': (.050, .008), 'FD': (.110, .028), 'FP': (.120, .032), 'PL': (.040, .007),
      'PS': (.048, .010), 'SE': (.020, .003), 'VS1': (.070, .015), 'VS2': (.085, .020), 'VS3': (.100, .025)}

truth = np.zeros((H, W), int)                                          # 10 blocos (2 x 5)
for k in range(10):
    r0, c0 = (k // 5) * 50, (k % 5) * 30
    truth[r0:r0 + 50, c0:c0 + 30] = k

def write(path, arr, dtype, nodata=None, desc=None):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, 'w', driver='GTiff', height=H, width=W, count=arr.shape[0], dtype=dtype, crs=CRS,
                       transform=TR, nodata=nodata) as d:
        d.write(arr.astype(dtype))
        for i, n in enumerate(desc or []):
            d.set_band_description(i + 1, n)
    return str(path)

def sar_image(order, year):
    tex = gaussian_filter(rng.normal(size=(H, W)), 1.6); tex = np.exp(0.5 * tex / tex.std())
    bands = {}
    for b, pol in enumerate(['HH', 'HV']):
        mean = np.array([MU[c][b] for c in CLASSES])[truth] * tex
        bands[pol] = rng.gamma(4.0, mean / 4.0)
    return write(OUT / f'SAR_{year}.tif', np.stack([bands[o] for o in order]), 'float32')   # SEM nomes de banda (como os originais)

samples_r = np.zeros((H, W), np.int16); polys, names = [], []
for k, c in enumerate(CLASSES):
    r0, c0 = (k // 5) * 50, (k % 5) * 30
    for _ in range(3):
        rr, cc = r0 + rng.integers(1, 30), c0 + rng.integers(1, 9)
        samples_r[rr:rr + 18, cc:cc + 18] = k + 1
        x0, y1 = TR * (cc, rr); x1, y0 = TR * (cc + 18, rr + 18)
        polys.append(box(x0, y0, x1, y1)); names.append(c)
files = {'sar_2008': sar_image(['HV', 'HH'], 2008), 'sar_2010': sar_image(['HH', 'HV'], 2010)}
for y in (2008, 2010):
    files[f'amostras_{y}'] = write(OUT / f'amostras_total_{y}_10classes.tif', samples_r, 'int16', nodata=0)
    files[f'poligonos_{y}'] = str(OUT / f'amostras_{y}.shp')
    gpd.GeoDataFrame({'Classe': names}, geometry=polys, crs=CRS).to_file(files[f'poligonos_{y}'])

# verossimilhanças ópticas (Gaussiana 1D por classe óptica) — densidades, sem descrição de banda
opt_truth = np.vectorize(lambda k: OPT_CLASSES.index(OPT[CLASSES[k]]))(truth)
mu_opt = np.linspace(0.05, 0.40, 6)
for y in (2007, 2009):
    x = rng.normal(mu_opt[opt_truth], 0.045)
    lik = np.stack([np.exp(-0.5 * ((x - m) / 0.045) ** 2) / (0.045 * np.sqrt(2 * np.pi)) for m in mu_opt])
    files[f'opt_{y}'] = write(OUT / f'likelihoods_{y}_resampled.tif', lik, 'float32')
cloud = np.zeros((H, W), np.uint8); cloud[60:85, 20:70] = 1
files['nuvem_2007'] = write(OUT / 'cloud_mask_2007.tif', cloud, 'uint8')
# polígonos de validação óptica (campo DN = posição da classe em OPT_CLASSES, 1-based)
vp, dn = [], []
for k in range(10):
    r0, c0 = (k // 5) * 50, (k % 5) * 30
    rr, cc = r0 + 32, c0 + 12
    x0, y1 = TR * (cc, rr); x1, y0 = TR * (cc + 14, rr + 14)
    vp.append(box(x0, y0, x1, y1)); dn.append(OPT_CLASSES.index(OPT[CLASSES[k]]) + 1)
for y in (2007, 2009):
    files[f'valid_{y}'] = str(OUT / f'test_samples_{y}.shp')
    gpd.GeoDataFrame({'DN': dn}, geometry=vp, crs=CRS).to_file(files[f'valid_{y}'])

# matrizes de transição (validade): mesma classe agregada, desmatamento (F/VS -> AG/PA/SE) e regeneração
group = lambda c: OPT.get(c, c)
def allowed(a, b):
    ga, gb = group(a), group(b)
    return ga == gb or (ga in ('F', 'VS1', 'VS3') and gb in ('AG', 'PA', 'SE')) or (ga in ('AG', 'PA', 'SE') and gb == 'VS1') \
        or (ga == 'VS1' and gb == 'VS3') or (ga in ('AG', 'PA', 'SE') and gb in ('AG', 'PA', 'SE'))
for (a_list, b_list, name) in [(OPT_CLASSES, CLASSES, '2007-2008'), (CLASSES, OPT_CLASSES, '2008-2009')]:
    m = pd.DataFrame([[1.0 if allowed(a, b) else 0.0 for b in b_list] for a in a_list], index=a_list, columns=b_list)
    files[f'trans_{name}'] = str(OUT / f'{name}.txt'); m.to_csv(files[f'trans_{name}'], sep=';')

# tabela de trajetórias PRONTA no formato do R (códigos = posição da banda; coluna final_weights)
import itertools
m1 = pd.read_csv(files['trans_2007-2008'], sep=';', index_col=0); m2 = pd.read_csv(files['trans_2008-2009'], sep=';', index_col=0)
rows = [{'X2007': i + 1, 'X2008': j + 1, 'X2009': k + 1, 'final_weights': m1.iloc[i, j] * m2.iloc[j, k]}
        for i, j, k in itertools.product(range(6), range(10), range(6))]
files['pesos'] = str(OUT / 'pesos_exemplo.txt'); pd.DataFrame(rows).to_csv(files['pesos'], sep=';', index=False)

cfg = {
    "raiz_resultados": str(OUT.parent / 'resultados'),
    "classes": CLASSES,
    "niveis_hierarquicos": {"lvl_1": [["VS1", "VS2", "VS3"]],
                            "lvl_2": [["FP", "FD"], ["VS1", "VS2", "VS3"], ["AP", "AC"], ["PL", "PS"]]},
    "legenda_final": {"AP+AC": "AG", "PL+PS": "PA", "VS1+VS2+VS3": "VS", "FP+FD": "F", "SE": "SE"},
    "imagens": {
        "2008": {"sensor": "SAR", "imagem": files['sar_2008'], "bandas": ["HV", "HH"],
                 "amostras": files['poligonos_2008'], "coluna_classe": "Classe", "nomes_classes": {}},
        "2010": {"sensor": "SAR", "imagem": files['sar_2010'], "bandas": ["HH", "HV"],
                 "amostras": files['amostras_2010']}},
    "cenarios": {
        "C2_par_intensidade": {
            "datas": {
                "2007": {"verossimilhanca": files['opt_2007'], "tipo": "direta", "classes_bandas": OPT_CLASSES,
                         "mascara": files['nuvem_2007'],
                         "validacao": {"arquivo": files['valid_2007'], "tipo": "poligonos", "coluna": "DN",
                                       "nomes": {str(i + 1): c for i, c in enumerate(OPT_CLASSES)}},
                         "agregacao": None},
                "2008": {"verossimilhanca": "{resultados}/2008/verossimilhanca/par_intensidade.tif", "tipo": "log",
                         "classes_bandas": None, "mascara": None,
                         "validacao": {"arquivo": "{resultados}/2008/amostras.gpkg", "camada": "validacao",
                                       "tipo": "pontos", "coluna": "class_name"},
                         "agregacao": "lvl_2"},
                "2009": {"verossimilhanca": files['opt_2009'], "tipo": "direta", "classes_bandas": OPT_CLASSES,
                         "mascara": None,
                         "validacao": {"arquivo": files['valid_2009'], "tipo": "poligonos", "coluna": "DN",
                                       "nomes": {str(i + 1): c for i, c in enumerate(OPT_CLASSES)}},
                         "agregacao": None}},
            "transicoes": [files['trans_2007-2008'], files['trans_2008-2009']],
            "transicoes_com_rotulos": True}}
}
import copy
cfg['cenarios']['C2_tabela_R'] = copy.deepcopy(cfg['cenarios']['C2_par_intensidade'])
cfg['cenarios']['C2_tabela_R']['transicoes'] = None
cfg['cenarios']['C2_tabela_R']['tabela_trajetorias'] = files['pesos']
(OUT.parent / 'config_exemplo.json').write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding='utf-8')
print('Projeto fictício em', OUT)
