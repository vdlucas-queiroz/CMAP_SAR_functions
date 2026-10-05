"""
Desenho amostral espacial (origem: 14_sample_design.R e SAMPLES_TO_POINTS.R).

Fluxo
-----
raster de amostras (códigos de classe)
   -> raster_samples_to_points      todos os pixels de amostra como pontos (centro do pixel)
   -> random_split_points           divisão aleatória estratificada (com balanceamento opcional)
      ou lag_grid_split             divisão sistemática por lags (pixels descorrelacionados no treino)
   -> extract_samples               valores da imagem nos pontos -> df_samples (coluna 'Class')
   -> save_points                   GeoPackage / Shapefile

Convenções
----------
- class_names: dict {código: nome} ou lista em que o código k corresponde a class_names[k-1]
  (convenção do R: classes 1..K).
- nodata_values: códigos ignorados (padrão 0, como no R), além do nodata declarado no raster.
- lags: DataFrame com colunas 'h' e 'v' indexado pelo NOME ou pelo CÓDIGO da classe
  (saída de autocorrelation.lags_for_sampling), ou dict {classe: (h, v)}.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .samples import CLASS_COL


# =============================================================================
# Utilidades
# =============================================================================
def _class_map(class_names, codes):
    if isinstance(class_names, dict):
        mapping = {int(k): str(v) for k, v in class_names.items()}
    else:
        mapping = {i + 1: str(n) for i, n in enumerate(class_names)}
    missing = sorted(set(int(c) for c in codes) - set(mapping))
    if missing:
        raise ValueError(f"Códigos sem nome em class_names: {missing}")
    return mapping


def _read_class_raster(raster_path, nodata_values):
    import rasterio
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        meta = {'transform': src.transform, 'crs': src.crs, 'height': src.height,
                'width': src.width, 'res': src.res, 'bounds': src.bounds}
        valid = np.ones(data.shape, bool)
        if src.nodata is not None:
            valid &= ~(data == src.nodata)
    if np.issubdtype(data.dtype, np.floating):
        valid &= np.isfinite(data)
    for v in nodata_values or ():
        valid &= data != v
    return data, valid, meta


def _points_from_cells(rows, cols, codes, mapping, meta):
    import geopandas as gpd
    from rasterio.transform import xy
    xs, ys = xy(meta['transform'], rows, cols, offset='center')
    return gpd.GeoDataFrame({'class_id': np.asarray(codes, int),
                             'class_name': [mapping[int(c)] for c in codes],
                             'row': np.asarray(rows, int), 'col': np.asarray(cols, int)},
                            geometry=gpd.points_from_xy(xs, ys), crs=meta['crs'])


# =============================================================================
# Raster de amostras -> pontos
# =============================================================================
def raster_samples_to_points(raster_path, class_names, nodata_values=(0,)):
    """
    Todos os pixels de amostra como pontos no centro do pixel (raster_to_points).
    Returns: GeoDataFrame com class_id, class_name, row, col, geometry (CRS do raster).
    """
    data, valid, meta = _read_class_raster(raster_path, nodata_values)
    rows, cols = np.nonzero(valid)
    codes = data[rows, cols]
    order = np.lexsort((np.ravel_multi_index((rows, cols), data.shape), codes))
    return _points_from_cells(rows[order], cols[order], codes[order], _class_map(class_names, codes), meta)


def random_split_points(points, proportion, balance=None, random_state=None):
    """
    Divisão aleatória estratificada (raster_to_shapefile_sample).

    Treino por classe: round(proportion x n) (arredondamento bancário, como round() do R).
    balance=None: sem balanceamento.
    balance=1   : treino limitado ao nº de pixels da menor classe.
    balance=b>1 : treino limitado a min(b x menor classe, n).
    A "menor classe" considera apenas classes válidas (o original incluía o código 0/nodata).

    Returns: (train, valid) GeoDataFrames.
    """
    rng = np.random.default_rng(random_state)
    counts = points['class_id'].value_counts()
    min_n = int(counts.min())
    train_idx = []
    for code in sorted(counts.index):
        idx = points.index[points['class_id'] == code].to_numpy()
        k = int(np.round(proportion * len(idx)))
        if balance is not None:
            k = min(k, min_n) if balance == 1 else min(k, min(min_n * balance, len(idx)))
        train_idx.extend(rng.choice(idx, size=int(k), replace=False))
    mask = points.index.isin(train_idx)
    return points[mask], points[~mask]


def _lag_for(lags, code, name):
    if isinstance(lags, dict):
        for key in (name, code, str(code)):
            if key in lags:
                return tuple(int(v) for v in lags[key])
    else:
        for key in (name, code, str(code)):
            if key in lags.index:
                return int(lags.loc[key, 'h']), int(lags.loc[key, 'v'])
    raise ValueError(f"Lag não informado para a classe {name} (código {code}).")


def lag_grid_split(raster_path, class_names, lags, nodata_values=(0, -9999)):
    """
    Divisão sistemática por lags (raster_to_shapefile_sample_by_lags).

    Para cada classe, uma grade regular com passo de h colunas e v linhas, começando no
    pixel do canto INFERIOR esquerdo (como no original, que parte de ymin):
        treino    = pixels da classe que caem na grade;
        validação = demais pixels da classe.
    Com h = v = 1, todos os pixels da classe vão para o treino (sem validação).

    Returns: (train, valid) GeoDataFrames.
    """
    data, valid, meta = _read_class_raster(raster_path, nodata_values)
    codes_present = np.unique(data[valid])
    mapping = _class_map(class_names, codes_present)
    n_rows, n_cols = data.shape
    train_parts, valid_parts = [], []
    for code in codes_present:
        name = mapping[int(code)]
        h, v = _lag_for(lags, int(code), name)
        cls = valid & (data == code)
        grid = np.zeros_like(cls)
        grid[(n_rows - 1)::-v, ::h] = True             # linhas a partir da última (ymin), colunas a partir da 1ª
        tr = cls & grid
        rows, cols = np.nonzero(tr)
        train_parts.append((rows, cols, np.full(rows.size, code)))
        if not (h == 1 and v == 1):
            rows, cols = np.nonzero(cls & ~grid)
            valid_parts.append((rows, cols, np.full(rows.size, code)))
        else:
            print(f"Classe {name}: h = v = 1 -> todos os pixels no treino (sem validação).")

    def build(parts):
        if not parts:
            return _points_from_cells([], [], [], mapping, meta)
        r, c, k = (np.concatenate(x) for x in zip(*parts))
        return _points_from_cells(r, c, k, mapping, meta)
    return build(train_parts), build(valid_parts)


def split_points(points, class_column, train_ratio, random_state=None):
    """
    Divide pontos já existentes em treino/validação por classe (divide_data):
    treino = floor(train_ratio x n) por classe, como dplyr::slice_sample(prop=).
    """
    if class_column not in points.columns:
        raise ValueError(f"A coluna '{class_column}' não existe nos pontos.")
    rng = np.random.default_rng(random_state)
    train_idx = []
    for _, g in points.groupby(class_column, sort=True):
        k = int(math.floor(train_ratio * len(g)))
        train_idx.extend(rng.choice(g.index.to_numpy(), size=k, replace=False))
    mask = points.index.isin(train_idx)
    return points[mask], points[~mask]


# =============================================================================
# Pontos -> conjunto amostral
# =============================================================================
def extract_samples(points, image_path, band_names=None, class_column='class_name', dropna=True):
    """
    Valores da imagem nos pontos -> df_samples (coluna 'Class' + uma coluna por banda).
    Os pontos são reprojetados para o CRS da imagem, se necessário.
    """
    import rasterio
    with rasterio.open(image_path) as src:
        pts = points.to_crs(src.crs) if (points.crs is not None and src.crs is not None
                                         and points.crs != src.crs) else points
        coords = [(g.x, g.y) for g in pts.geometry]
        values = np.array(list(src.sample(coords)), dtype=np.float64)
        nodata = src.nodata
        names = band_names or [d if d else f'band_{i + 1}' for i, d in enumerate(src.descriptions)]
    if nodata is not None:
        values[values == nodata] = np.nan
    df = pd.DataFrame(values, columns=names, index=points.index)
    df.insert(0, CLASS_COL, points[class_column].to_numpy())
    if dropna:
        n0 = len(df)
        df = df.dropna()
        if len(df) < n0:
            print(f"extract_samples: {n0 - len(df)} pontos sem valor válido na imagem foram descartados.")
    return df


def save_points(points, path, layer=None):
    """Grava pontos (GeoPackage recomendado; .shp aceito). Cria a pasta, se necessário."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = 'GPKG' if path.suffix.lower() == '.gpkg' else None
    points.to_file(path, layer=layer, driver=driver)
    return str(path)


def polygon_envelopes(polygons):
    """Retângulo envolvente de cada polígono (bounding_box do notebook de correlação)."""
    out = polygons.copy()
    out['geometry'] = polygons.geometry.envelope
    return out


def extract_polygon_samples(polygons, image_path, class_column, band_names=None, dropna=True):
    """
    Todos os pixels cujo CENTRO está dentro dos polígonos (regra do raster::extract do R),
    com a classe do polígono -> df_samples. Usado para amostras de validação em polígonos.
    Pixels cobertos por polígonos de classes diferentes são descartados (com aviso).
    """
    import rasterio
    from rasterio.features import rasterize
    with rasterio.open(image_path) as src:
        polys = polygons.to_crs(src.crs) if (polygons.crs is not None and src.crs is not None
                                            and polygons.crs != src.crs) else polygons
        names = sorted(polys[class_column].astype(str).unique())
        code = {n: i + 1 for i, n in enumerate(names)}
        label = np.zeros((src.height, src.width), np.int32)
        count = np.zeros((src.height, src.width), np.int32)
        for n in names:
            geoms = list(polys.loc[polys[class_column].astype(str) == n, 'geometry'])
            m = rasterize([(g, 1) for g in geoms], out_shape=label.shape, transform=src.transform,
                          fill=0, all_touched=False, dtype='uint8').astype(bool)
            label[m] = code[n]
            count += m
        conflict = count > 1
        if conflict.any():
            print(f"extract_polygon_samples: {int(conflict.sum())} pixels em polígonos de classes diferentes descartados.")
        rows, cols = np.nonzero((label > 0) & ~conflict)
        data = src.read().astype(np.float64)[:, rows, cols].T
        nodata = src.nodata
        bn = band_names or [d if d else f'band_{i + 1}' for i, d in enumerate(src.descriptions)]
    if nodata is not None:
        data[data == nodata] = np.nan
    inv = {v: k for k, v in code.items()}
    df = pd.DataFrame(data, columns=bn)
    df.insert(0, CLASS_COL, [inv[c] for c in label[rows, cols]])
    df['row'], df['col'] = rows, cols
    if dropna:
        df = df.dropna(subset=bn)
    return df
