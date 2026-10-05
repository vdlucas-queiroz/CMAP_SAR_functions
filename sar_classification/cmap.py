"""
Classificador CMAP (Eq. 2.11), integrado ao pacote (origem: CMAP_FUNCTIONS.ipynb e CMAP_C4_C5.R).

    ŝ = arg max_s  log P(s) + Σ_t log P(o_t | ω_t)

Recursos
--------
- bands_config=None: classes de cada banda lidas da descrição das bandas;
- input_is_log: um valor para todas as datas ou uma lista por data, permitindo misturar
  verossimilhanças diretas (ex.: ópticas) e log-verossimilhanças (ex.: SAR, likelihood_raster(log=True));
- masks: máscara por data (1 = nuvem; None = sem máscara), com dois modos:
    mask_mode='output' — como o CMAP_classifier do R: a verossimilhança da data nublada é usada
                         normalmente e o pixel só é zerado na saída daquela data;
    mask_mode='ignore' — a data nublada não informa a trajetória naquele pixel (log-verossimilhança
                         constante); o pixel é classificado pelas demais datas e zerado na saída
                         daquela data;
- class_codes: codificação de saída comum ao MaxVer;
- output_layout='stack': UM GeoTIFF multibanda (uma banda por data) em vez de um arquivo por data.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def read_bands_config(tif_paths):
    """Classes de cada banda, lidas da descrição das bandas de cada raster."""
    import rasterio
    cfg = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            d = list(src.descriptions)
        if any(x is None or x == '' for x in d):
            raise ValueError(f"'{p}' tem bandas sem descrição; informe bands_config.")
        cfg.append(d)
    return cfg


def default_class_codes(class_names):
    """Códigos 1..K em ordem alfabética (convenção do CMAP_FUNCTIONS)."""
    return {c: i + 1 for i, c in enumerate(sorted(class_names))}


def _per_date(value, n, name):
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"{name} tem {len(value)} elementos; esperado {n}.")
        return list(value)
    return [value] * n


def _read_loglik(path, n_bands, is_log):
    """Lê um raster de (log-)verossimilhança -> (log-verossimilhança float32, máscara de pixels inválidos)."""
    import rasterio
    with rasterio.open(path) as src:
        if src.count != n_bands:
            raise ValueError(f"'{path}' tem {src.count} bandas; esperado {n_bands}.")
        data = src.read().astype(np.float64)
        bad = ~np.isfinite(data) if not is_log else np.isnan(data)
        if src.nodata is not None and not np.isnan(src.nodata):
            bad |= data == src.nodata
        meta = {'height': src.height, 'width': src.width, 'crs': src.crs, 'transform': src.transform}
    if is_log:
        ll = data
    else:
        bad |= data < 0
        with np.errstate(divide='ignore', invalid='ignore'):
            ll = np.log(data)
    ll[bad] = -np.inf
    return ll.astype(np.float32), bad.any(axis=0), meta


def _read_mask(path, shape):
    import rasterio
    if path is None:
        return np.zeros(shape, bool)
    with rasterio.open(path) as src:
        m = src.read(1)
        if m.shape != shape:
            raise ValueError(f"Máscara '{path}' com dimensões {m.shape}; esperado {shape}.")
        return m == 1


def _write_outputs(arrs, names, meta, output_dir, layout, file_name, dtype, nodata):
    import rasterio
    prof = {'driver': 'GTiff', 'height': meta['height'], 'width': meta['width'], 'crs': meta['crs'],
            'transform': meta['transform'], 'compress': 'deflate', 'dtype': dtype, 'nodata': nodata}
    paths = []
    if layout == 'stack':
        p = output_dir / f'{file_name}.tif'
        with rasterio.open(p, 'w', count=len(arrs), **prof) as dst:
            for i, (a, n) in enumerate(zip(arrs, names), start=1):
                dst.write(a.astype(dtype), i)
                dst.set_band_description(i, str(n))
        paths.append(str(p))
    else:
        for a, n in zip(arrs, names):
            p = output_dir / f'{file_name}_{n}.tif'
            with rasterio.open(p, 'w', count=1, **prof) as dst:
                dst.write(a.astype(dtype), 1)
                dst.set_band_description(1, str(n))
            paths.append(str(p))
    return paths


def cmap_classifier(tif_paths, weights_df, bands_config=None, output_dir='cmap_output', date_labels=None,
                    class_codes=None, input_is_log=False, masks=None, mask_mode='output',
                    output_layout='per_date', save_trajectory_ids=True, save_log_posterior=True,
                    progress_step=10, verbose=True):
    """
    Returns: dict com classified (caminho(s)), trajectory_ids, log_posterior, legend, trajectory_keys,
             class_codes e class_counts.
    output_layout: 'per_date' (Classified_<data>.tif, como no CMAP_FUNCTIONS) ou 'stack' (CMAP.tif multibanda).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_dates = len(tif_paths)
    bands_config = read_bands_config(tif_paths) if bands_config is None else bands_config
    if len(bands_config) != n_dates:
        raise ValueError(f"bands_config tem {len(bands_config)} datas; foram informados {n_dates} rasters.")
    date_labels = [str(d) for d in (date_labels or range(1, n_dates + 1))]
    is_log = _per_date(input_is_log, n_dates, 'input_is_log')
    masks = _per_date(masks, n_dates, 'masks')
    if mask_mode not in ('output', 'ignore'):
        raise ValueError("mask_mode deve ser 'output' ou 'ignore'.")
    stage_cols = [c for c in weights_df.columns if str(c).startswith('Stage_')]
    if len(stage_cols) != n_dates:
        raise ValueError(f"A tabela tem {len(stage_cols)} estágios; foram informados {n_dates} rasters.")

    bands_config = [[str(c).strip() for c in cfg] for cfg in bands_config]
    names = weights_df[stage_cols].astype(str).apply(lambda s: s.str.strip()).to_numpy()
    weights = weights_df['final_weight'].to_numpy(dtype=np.float64)
    n_traj = len(weights_df)
    classes = sorted(set(names.ravel()))
    class_codes = default_class_codes(classes) if class_codes is None else {str(k): int(v) for k, v in class_codes.items()}
    miss = set(classes) - set(class_codes)
    if miss:
        raise ValueError(f"class_codes não contém as classes: {sorted(miss)}")
    if max(class_codes.values()) > 255 or min(class_codes.values()) < 1:
        raise ValueError("class_codes deve usar valores 1..255 (0 = nodata).")

    band_lookup = np.full((n_traj, n_dates), -1, dtype=np.int32)
    code_lookup = np.zeros((n_traj, n_dates), dtype=np.uint8)
    for d in range(n_dates):
        pos = {c: i for i, c in enumerate(bands_config[d])}
        band_lookup[:, d] = [pos.get(c, -1) for c in names[:, d]]
        code_lookup[:, d] = [class_codes[c] for c in names[:, d]]
        absent = sorted(set(names[:, d]) - set(pos))
        if absent and verbose:
            print(f"Aviso: classes sem banda na data {date_labels[d]}: {absent} -> P = 0.")
    valid_traj = (weights > 0) & (band_lookup >= 0).all(axis=1)
    if not valid_traj.any():
        raise ValueError("Nenhuma trajetória válida.")
    with np.errstate(divide='ignore'):
        log_w = np.log(weights)

    log_lik, cloud, valid_px, meta = [], [], None, None
    for d, path in enumerate(tif_paths):
        ll, bad, m = _read_loglik(path, len(bands_config[d]), is_log[d])
        if meta is None:
            meta = m
        elif (m['height'], m['width']) != (meta['height'], meta['width']):
            raise ValueError(f"'{path}' tem dimensões diferentes do primeiro raster.")
        shape = (meta['height'], meta['width'])
        cl = _read_mask(masks[d], shape)
        if mask_mode == 'ignore':
            ll[:, cl] = 0.0                                # data não informativa nesse pixel
            bad = bad & ~cl
        valid_px = ~bad if valid_px is None else valid_px & ~bad
        log_lik.append(ll)
        cloud.append(cl)
    h, w = meta['height'], meta['width']

    max_lp = np.full((h, w), -np.inf)
    best = np.full((h, w), -1, dtype=np.int32)
    idx_valid = np.flatnonzero(valid_traj)
    next_rep = progress_step
    for k, s in enumerate(idx_valid):
        acc = np.full((h, w), log_w[s])
        for d in range(n_dates):
            acc += log_lik[d][band_lookup[s, d]]
        better = acc > max_lp
        max_lp[better] = acc[better]
        best[better] = s
        pct = 100 * (k + 1) / len(idx_valid)
        if verbose and pct >= next_rep:
            print(f"  Progresso: {pct:5.1f}% ({k + 1}/{len(idx_valid)})")
            next_rep += progress_step
    best[~valid_px] = -1
    ok = best >= 0
    safe = np.where(ok, best, 0)

    maps = [np.where(ok & ~cloud[d], code_lookup[safe, d], 0).astype(np.uint8) for d in range(n_dates)]
    out = {'class_codes': class_codes}
    out['classified'] = _write_outputs(maps, date_labels, meta, output_dir, output_layout,
                                       'CMAP' if output_layout == 'stack' else 'Classified', 'uint8', 0)
    if output_layout == 'stack':
        out['classified'] = out['classified'][0]
    if save_trajectory_ids:
        out['trajectory_ids'] = _write_outputs([best], ['Trajectory_ID'], meta, output_dir, 'stack', 'Trajectory_IDs', 'int32', -1)[0]
    if save_log_posterior:
        out['log_posterior'] = _write_outputs([np.where(ok, max_lp, np.nan)], ['log_posterior'], meta, output_dir,
                                              'stack', 'Log_Posterior', 'float32', np.nan)[0]
    legend = pd.DataFrame(sorted(class_codes.items(), key=lambda x: x[1]), columns=['Class', 'ID'])
    out['legend_table'] = legend
    if output_layout == 'per_date':                     # compatibilidade com o CMAP_FUNCTIONS
        p = output_dir / 'legend_keys.csv'
        legend.to_csv(p, sep=';', index=False)
        out['legend'] = str(p)

    inv = {v: k for k, v in class_codes.items()}
    counts = {}
    for d in range(n_dates):
        v, c = np.unique(maps[d][maps[d] > 0], return_counts=True)
        counts[date_labels[d]] = {inv[int(a)]: int(b) for a, b in zip(v, c)}
    cc = pd.DataFrame(counts).reindex(classes).fillna(0).astype(int)
    cc.index.name = 'Class'
    out['class_counts'] = cc
    ids, n_pix = np.unique(best[ok], return_counts=True)
    keys = weights_df[stage_cols + ['final_weight']].copy()
    keys.insert(0, 'Trajectory_ID', np.arange(n_traj))
    keys['n_pixels'] = 0
    keys.loc[ids, 'n_pixels'] = n_pix
    out['trajectory_table'] = keys
    if output_layout == 'per_date':                     # compatibilidade com o CMAP_FUNCTIONS
        keys.to_csv(output_dir / 'trajectory_keys.csv', sep=';', index=False)
        cc.to_csv(output_dir / 'class_counts.csv', sep=';')
        out['trajectory_keys'] = str(output_dir / 'trajectory_keys.csv')
    if verbose:
        print(f"Pixels válidos: {int(valid_px.sum())} de {h * w} | classificados: {int(ok.sum())}")
        print(f"Saídas em: {output_dir}")
    return out


def maxver_from_likelihoods(tif_paths, output_path, class_codes, bands_config=None, date_labels=None,
                            input_is_log=False, masks=None):
    """
    MaxVer de cada data a partir dos MESMOS rasters de verossimilhança usados no CMAP
    (equivale ao CMAP com todas as transições permitidas, como na dissertação) -> UM GeoTIFF
    uint8 multibanda (uma banda por data; 0 = nodata ou máscara).
    """
    n = len(tif_paths)
    bands_config = read_bands_config(tif_paths) if bands_config is None else bands_config
    date_labels = [str(d) for d in (date_labels or range(1, n + 1))]
    is_log, masks = _per_date(input_is_log, n, 'input_is_log'), _per_date(masks, n, 'masks')
    maps, meta = [], None
    for d, path in enumerate(tif_paths):
        ll, bad, m = _read_loglik(path, len(bands_config[d]), is_log[d])
        meta = meta or m
        lut = np.array([class_codes[str(c).strip()] for c in bands_config[d]], dtype=np.uint8)
        arg = np.argmax(ll, axis=0)
        cl = _read_mask(masks[d], arg.shape)
        maps.append(np.where(bad | cl, 0, lut[arg]).astype(np.uint8))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _write_outputs(maps, date_labels, meta, output_path.parent, 'stack', output_path.stem, 'uint8', 0)[0]
