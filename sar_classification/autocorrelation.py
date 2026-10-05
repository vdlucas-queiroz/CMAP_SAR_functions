"""
Autocorrelação espacial das amostras e lags de descorrelação (origem: correlation_2.ipynb).

Fluxo
-----
polígonos de amostra + imagem
   -> sample_lags          lag horizontal e vertical de cada polígono, por banda
   -> summarize_lags       mediana e máximo sem outliers (IQR), por classe e banda
   -> lags_for_sampling    maior lag entre as bandas, por classe -> sampling.lag_grid_split

Definição (por polígono, com média x̄ e variância s² dos pixels válidos):
    R(dx, dy) = [ Σ (x_p - x̄)(x_q - x̄) / N(dx, dy) ] / s²
sobre os pares (p, q) de pixels válidos separados por (dx, dy). O lag de descorrelação é o
primeiro lag em que R < target_correlation.
"""
import numpy as np
import pandas as pd


# =============================================================================
# Perfis de autocorrelação
# =============================================================================
def autocorrelation_profiles(array, valid=None, max_lag=20, estimator='pairs'):
    """
    Perfis de autocorrelação horizontal (dx = 0..k1-1, dy = 0) e vertical (dx = 0, dy = 0..k2-1).

    array: 2D; valid: máscara de pixels válidos (padrão: finitos e >= 0, como no original).
    k1 = min(n_col//3 + 2, max_lag); k2 = min(n_row//3 + 2, max_lag)  (regra do original).
    estimator:
        'pairs' (código original): R(s) = [C(s)/N_pares(s)] / [C(0)/N_validos]
        'total' (Eq. 4.3 da dissertação, Yanasse 1991): R(s) = γ(s)/γ(0) com γ(s) = C(s)/N,
                ou seja R(s) = C(s)/C(0) — decai mais rápido nos lags grandes.

    Returns: (R_h, R_v, N_h, N_v) — correlações e nº de pares por lag.
    """
    if estimator not in ('pairs', 'total'):
        raise ValueError("estimator deve ser 'pairs' ou 'total'.")
    x = np.asarray(array, dtype=np.float64)
    if valid is None:
        with np.errstate(invalid='ignore'):
            valid = np.isfinite(x) & (x >= 0)
    n_row, n_col = x.shape
    k1 = min(n_col // 3 + 2, max_lag)
    k2 = min(n_row // 3 + 2, max_lag)

    mean = x[valid].mean()
    d = np.where(valid, x - mean, 0.0)
    w = valid.astype(np.float64)

    def profile(k, axis):
        c, n = np.zeros(k), np.zeros(k)
        for lag in range(k):
            if axis == 1:
                a, b, va, vb = d[:, :n_col - lag], d[:, lag:], w[:, :n_col - lag], w[:, lag:]
            else:
                a, b, va, vb = d[:n_row - lag, :], d[lag:, :], w[:n_row - lag, :], w[lag:, :]
            c[lag], n[lag] = np.sum(a * b), np.sum(va * vb)
        return c, n

    ch, nh = profile(k1, axis=1)
    cv, nv = profile(k2, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        if estimator == 'total':
            return ch / ch[0], cv / ch[0], nh, nv
        var = ch[0] / nh[0]
        return ch / (nh * var), cv / (nv * var), nh, nv


def decorrelation_lag(profile, target_correlation, inclusive=False):
    """
    Primeiro lag (>= 0) com correlação abaixo do limiar; None se não houver.
    inclusive=False: R < limiar (código original); inclusive=True: R <= limiar (texto da dissertação).
    """
    p = np.asarray(profile)
    with np.errstate(invalid='ignore'):
        below = np.flatnonzero(p <= target_correlation if inclusive else p < target_correlation)
    return int(below[0]) if below.size else None


def plot_profiles(R_h, R_v, target_correlation, lag_h=None, lag_v=None, title='', pdf=None):
    """Perfis espelhados (lags negativos e positivos), como no original."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, prof, lag, name in [(axes[0], R_h, lag_h, 'Horizontal'), (axes[1], R_v, lag_v, 'Vertical')]:
        lags = np.arange(1 - len(prof), len(prof))
        ax.plot(lags, np.concatenate([prof[::-1], prof[1:]]), 'x-', color='k')
        ax.axhline(target_correlation, color='r', ls='--')
        if lag is not None:
            ax.axvline(lag, color='orange', ls='--')
        ax.set_title(f'{title} — {name}')
        ax.set_xlabel('Lag (pixels)')
        ax.set_ylabel('Autocorrelação')
    plt.tight_layout()
    if pdf is not None:
        pdf.savefig(plt.gcf(), bbox_inches='tight')
    plt.show()


# =============================================================================
# Polígonos -> lags
# =============================================================================
def polygon_window(src, polygon, band, value_transform='sqrt'):
    """
    Recorte do retângulo envolvente do polígono; pixels fora do polígono ou nodata -> NaN.
    value_transform: 'sqrt' (intensidade -> amplitude, como no original), None ou função.
    """
    from rasterio.features import geometry_mask
    from rasterio.windows import Window, from_bounds
    window = from_bounds(*polygon.bounds, transform=src.transform)
    rows, cols = window.toslices()                     # piso/teto: todos os pixels tocados pelo retângulo
    window = Window.from_slices(rows, cols).intersection(Window(0, 0, src.width, src.height))
    arr = src.read(band, window=window, boundless=False).astype(np.float64)
    if arr.size == 0:
        return arr
    inside = geometry_mask([polygon], out_shape=arr.shape, transform=src.window_transform(window), invert=True)
    bad = ~inside
    if src.nodata is not None:
        bad |= arr == src.nodata
    arr[bad] = np.nan
    if value_transform == 'sqrt':
        with np.errstate(invalid='ignore'):
            arr = np.sqrt(arr)
    elif callable(value_transform):
        arr = value_transform(arr)
    return arr


def sample_lags(raster_path, polygons, class_column, bands=None, target_correlation=0.3,
                value_transform='sqrt', max_lag=20, estimator='pairs', inclusive=False, plot=False,
                verbose=True):
    """
    Lag horizontal e vertical de descorrelação de cada polígono, em cada banda.

    Returns: DataFrame (polygon, class, band, lag_h, lag_v, n_valid).
    Polígonos inválidos ou sem lag abaixo do limiar recebem NaN.
    """
    import rasterio
    rows = []
    with rasterio.open(raster_path) as src:
        bands = bands or list(range(1, src.count + 1))
        polys = polygons.to_crs(src.crs) if polygons.crs is not None and polygons.crs != src.crs else polygons
        for band in bands:
            for idx, row in polys.iterrows():
                cls, geom = row[class_column], row.geometry
                rec = {'polygon': idx, 'class': cls, 'band': band, 'lag_h': np.nan, 'lag_v': np.nan, 'n_valid': 0}
                if geom is None or not geom.is_valid:
                    if verbose:
                        print(f"Polígono {idx} ({cls}) inválido: ignorado.")
                    rows.append(rec)
                    continue
                arr = polygon_window(src, geom, band, value_transform)
                valid = np.isfinite(arr) & (arr >= 0) if arr.size else np.zeros(0, bool)
                rec['n_valid'] = int(valid.sum())
                if rec['n_valid'] >= 2:
                    R_h, R_v, _, _ = autocorrelation_profiles(arr, valid, max_lag, estimator)
                    lh = decorrelation_lag(R_h, target_correlation, inclusive)
                    lv = decorrelation_lag(R_v, target_correlation, inclusive)
                    rec['lag_h'] = np.nan if lh is None else lh
                    rec['lag_v'] = np.nan if lv is None else lv
                    if plot:
                        plot_profiles(R_h, R_v, target_correlation, lh, lv, f'Polígono {idx} — {cls} (banda {band})')
                rows.append(rec)
    out = pd.DataFrame(rows)
    if verbose:
        print(f"{len(out)} recortes (polígonos x bandas); sem lag definido: "
              f"h = {int(out['lag_h'].isna().sum())}, v = {int(out['lag_v'].isna().sum())}.")
    return out


def max_without_outliers(values, iqr=True):
    """Máximo dos valores dentro de [Q1 - 1,5 IQR, Q3 + 1,5 IQR] (ou máximo simples)."""
    v = np.asarray(pd.Series(values).dropna(), float)
    if v.size == 0:
        return np.nan
    if not iqr:
        return float(v.max())
    q1, q3 = np.percentile(v, [25, 75])
    keep = v[(v >= q1 - 1.5 * (q3 - q1)) & (v <= q3 + 1.5 * (q3 - q1))]
    return float(keep.max()) if keep.size else np.nan


def summarize_lags(lags, iqr=True):
    """Por classe e banda: nº de polígonos, mediana e máximo sem outliers de lag_h e lag_v."""
    rows = []
    for (cls, band), g in lags.groupby(['class', 'band'], sort=True):
        rows.append({'class': cls, 'band': band, 'n_h': int(g['lag_h'].notna().sum()),
                     'n_v': int(g['lag_v'].notna().sum()),
                     'median_h': g['lag_h'].median(), 'median_v': g['lag_v'].median(),
                     'max_h': max_without_outliers(g['lag_h'], iqr), 'max_v': max_without_outliers(g['lag_v'], iqr)})
    return pd.DataFrame(rows)


def lags_for_sampling(summary, statistic='max'):
    """
    Lag final por classe = maior valor entre as bandas (statistic 'max' ou 'median').
    Returns: DataFrame indexado pela classe, colunas 'h' e 'v' (inteiros) -> sampling.lag_grid_split.
    """
    cols = {'max': ('max_h', 'max_v'), 'median': ('median_h', 'median_v')}[statistic]
    out = summary.groupby('class')[list(cols)].max()
    out.columns = ['h', 'v']
    missing = out.index[out.isna().any(axis=1)].tolist()
    if missing:
        raise ValueError(f"Classes sem lag válido em nenhuma banda: {missing}")
    return np.ceil(out).astype(int)
