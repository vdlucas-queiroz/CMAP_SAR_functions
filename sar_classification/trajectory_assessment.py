"""
Avaliação de trajetórias (origem: trajectory_assessment.R).

impossible_trajectories   — % de pixels cuja sequência de classes (ex.: MaxVer empilhado) é uma
                            trajetória impossível segundo as matrizes de transição (P(s) = 0).
compare_classifications   — discordância pixel a pixel entre duas séries classificadas
                            (ex.: MaxVer × CMAP), por data e acumulada, com máscara de nuvens.

Diferenças em relação ao R:
- a trajetória é avaliada transição a transição (impossível se algum M_t[c_t, c_t+1] = 0, o que
  equivale a P(s) = 0), sem a codificação 10^(6-i) do original — que limitava a série a 6 datas
  e os códigos de classe a um dígito;
- a comparação usa as legendas para traduzir os códigos em nomes de classe, de modo que MaxVer e
  CMAP podem usar codificações diferentes;
- os totais são calculados numericamente (o original comparava rownames como texto).
"""
import numpy as np
import pandas as pd


def expand_series(series):
    """
    Uma série classificada pode ser: lista de caminhos (uma banda cada), lista de (caminho, banda)
    ou UM caminho multibanda (uma banda por data). Returns: lista de (caminho, banda).
    """
    import rasterio
    if isinstance(series, (str, bytes)) or hasattr(series, '__fspath__'):
        with rasterio.open(series) as src:
            return [(series, b) for b in range(1, src.count + 1)]
    return [s if isinstance(s, tuple) else (s, 1) for s in series]


def _read_codes(path):
    import rasterio
    path, band = path if isinstance(path, tuple) else (path, 1)
    with rasterio.open(path) as src:
        a = src.read(band)
        valid = np.ones(a.shape, bool) if src.nodata is None else (a != src.nodata)
        if np.issubdtype(a.dtype, np.floating):
            valid &= np.isfinite(a)
        meta = {'crs': src.crs, 'transform': src.transform, 'height': src.height, 'width': src.width}
    return a, valid, meta


def _to_names(codes, valid, legend):
    names = np.full(codes.shape, None, dtype=object)
    if legend is None:
        names[valid] = codes[valid].astype(np.int64).astype(str)
    else:
        lut = {int(k): str(v) for k, v in legend.items()}
        flat = codes[valid].astype(np.int64)
        names[valid] = [lut.get(int(c)) for c in flat]
    return names


def _write(arr, meta, path, dtype, nodata):
    import rasterio
    prof = {'driver': 'GTiff', 'count': 1, 'dtype': dtype, 'nodata': nodata, 'compress': 'deflate', **meta}
    with rasterio.open(path, 'w', **prof) as dst:
        dst.write(arr.astype(dtype), 1)
    return str(path)


def impossible_trajectories(classified_paths, matrices, legends=None, mask_path=None, output_path=None,
                            exclude_codes=None, top=10):
    """
    classified_paths: rasters classificados, em ordem temporal (um a mais que o nº de matrizes).
    matrices        : DataFrames de transitions.read_transition_matrix (com os nomes das classes).
    legends         : {código: classe} único ou um por data; None -> o código é o nome
                      (compatível com matrizes sem rótulo, has_labels=False).
    mask_path       : máscara (1 = excluir), opcional — o original não usava máscara aqui.
    exclude_codes   : códigos tratados como nodata além do nodata do raster (ex.: [0]).

    Returns: dict percentage, n_valid, n_impossible, n_unknown, raster (se output_path) e
             table (trajetórias impossíveis mais frequentes).
    """
    classified_paths = expand_series(classified_paths)
    T = len(classified_paths)
    if len(matrices) != T - 1:
        raise ValueError(f"{T} datas exigem {T - 1} matrizes; recebidas {len(matrices)}.")
    legends = legends if isinstance(legends, list) else [legends] * T
    stacks, valid, meta = [], None, None
    for t, p in enumerate(classified_paths):
        a, v, m = _read_codes(p)
        if exclude_codes:
            v &= ~np.isin(a, exclude_codes)
        valid = v if valid is None else valid & v
        meta = meta or m
        stacks.append(a)
    if mask_path is not None:
        mk, mv, _ = _read_codes(mask_path)
        valid &= ~(mv & (mk == 1))
    names = [_to_names(a, valid, legends[t]) for t, a in enumerate(stacks)]

    impossible = np.zeros(valid.shape, bool)
    unknown = np.zeros(valid.shape, bool)
    for t, M in enumerate(matrices):
        r = M.index.get_indexer(pd.Series(names[t][valid]).astype(object))
        c = M.columns.get_indexer(pd.Series(names[t + 1][valid]).astype(object))
        known = (r >= 0) & (c >= 0)
        zero = np.zeros(r.size, bool)
        zero[known] = M.to_numpy()[r[known], c[known]] == 0
        impossible[valid] |= zero
        unknown[valid] |= ~known

    n_valid, n_imp = int(valid.sum()), int(impossible.sum())
    out = {'percentage': 100.0 * n_imp / n_valid if n_valid else np.nan, 'n_valid': n_valid,
           'n_impossible': n_imp, 'n_unknown': int(unknown.sum())}
    if out['n_unknown']:
        print(f"Aviso: {out['n_unknown']} pixels com classe ausente das matrizes (contados como possíveis, como no R).")
    if output_path is not None:
        res = np.where(valid, impossible.astype(np.uint8), 255)
        out['raster'] = _write(res, meta, output_path, 'uint8', 255)
    seqs = pd.DataFrame({f'Stage_{t + 1}': names[t][impossible] for t in range(T)})
    out['table'] = (seqs.value_counts().rename('n_pixels').reset_index().head(top) if n_imp else seqs)
    return out


def compare_classifications(paths_a, paths_b, mask_path=None, legend_a=None, legend_b=None,
                            date_labels=None, output_path=None, verbose=True):
    """
    Discordância entre duas séries classificadas (compareRasterClassification).

    Por data: % de pixels com classes diferentes, entre os pixels válidos nas duas séries e
    fora da máscara (máscara = 1 é excluída).
    Acumulado: nº de datas com discordância por pixel; total = % de pixels com >= 1 discordância,
    entre os pixels válidos em todas as datas e fora da máscara.
    Com legendas ({código: classe}), a comparação é feita pelo NOME da classe.

    Returns: dict per_date (DataFrame), total_percentage, histogram (nº de discordâncias -> pixels),
             accumulated (array; -1 = máscara, -9999 = inválido) e raster (se output_path).
    """
    paths_a, paths_b = expand_series(paths_a), expand_series(paths_b)
    if len(paths_a) != len(paths_b):
        raise ValueError("As duas séries devem ter o mesmo nº de datas.")
    if legend_a is None and legend_b is None and verbose:
        print("Aviso: sem legendas — os CÓDIGOS são comparados diretamente; "
              "confirme que as duas séries usam a mesma codificação.")
    T = len(paths_a)
    date_labels = date_labels or [str(i + 1) for i in range(T)]
    cloud = None
    if mask_path is not None:
        cloud = None
        for mp in (mask_path if isinstance(mask_path, list) else [mask_path]):
            if mp is None:
                continue
            mk, mv, _ = _read_codes(mp)
            c = mv & (mk == 1)
            cloud = c if cloud is None else cloud | c

    rows, acc, all_valid, meta = [], None, None, None
    for t in range(T):
        a, va, meta = _read_codes(paths_a[t])
        b, vb, _ = _read_codes(paths_b[t])
        if a.shape != b.shape:
            raise ValueError(f"Data {date_labels[t]}: rasters com dimensões diferentes.")
        valid = va & vb
        if cloud is not None:
            valid &= ~cloud
        na, nb = _to_names(a, valid, legend_a), _to_names(b, valid, legend_b)
        diff = np.zeros(a.shape, np.int16)
        diff[valid] = (na[valid] != nb[valid]).astype(np.int16)
        acc = diff if acc is None else acc + diff
        all_valid = valid if all_valid is None else all_valid & valid
        n, k = int(valid.sum()), int(diff[valid].sum())
        rows.append({'date': date_labels[t], 'n_valid': n, 'n_disagree': k, 'percentage': 100.0 * k / n if n else np.nan})

    accumulated = np.where(all_valid, acc, -9999).astype(np.int16)
    if cloud is not None:
        accumulated[cloud] = -1
    vals = accumulated[all_valid]
    total = 100.0 * np.count_nonzero(vals > 0) / vals.size if vals.size else np.nan
    hist = pd.Series(vals).value_counts().sort_index().rename('n_pixels')
    hist.index.name = 'n_disagreements'
    out = {'per_date': pd.DataFrame(rows), 'total_percentage': total, 'histogram': hist, 'accumulated': accumulated}
    if output_path is not None:
        out['raster'] = _write(accumulated, meta, output_path, 'int16', -9999)
    if verbose:
        print(out['per_date'].round(3).to_string(index=False))
        print(f"Discordância total (>= 1 data): {total:.3f}%")
    return out


def impossible_from_table(classified_paths, trajectory_table, legends=None, exclude_codes=None, top=10):
    """
    Trajetórias impossíveis consultando diretamente uma tabela de trajetórias
    (Stage_1..Stage_T, final_weight — de transitions.trajectory_priors ou read_trajectory_table):
    o pixel é impossível se a sua sequência de classes tem final_weight = 0 (ou não consta da tabela).
    Equivale a impossible_trajectories quando a tabela vem do produto das matrizes.
    """
    classified_paths = expand_series(classified_paths)
    T = len(classified_paths)
    legends = legends if isinstance(legends, list) else [legends] * T
    stacks, valid = [], None
    for p in classified_paths:
        a, v, _ = _read_codes(p)
        if exclude_codes:
            v &= ~np.isin(a, exclude_codes)
        valid = v if valid is None else valid & v
        stacks.append(a)
    names = pd.DataFrame({f'Stage_{t + 1}': _to_names(a, valid, legends[t])[valid] for t, a in enumerate(stacks)})
    stage_cols = list(names.columns)
    tab = trajectory_table[stage_cols + ['final_weight']].astype({c: str for c in stage_cols})
    merged = names.astype(str).merge(tab, on=stage_cols, how='left')
    unknown = merged['final_weight'].isna()
    impossible = (merged['final_weight'].fillna(0) == 0).to_numpy()
    n_valid, n_imp = int(valid.sum()), int(impossible.sum())
    out = {'percentage': 100.0 * n_imp / n_valid if n_valid else np.nan, 'n_valid': n_valid,
           'n_impossible': n_imp, 'n_unknown': int(unknown.sum())}
    if out['n_unknown']:
        print(f"Aviso: {out['n_unknown']} pixels com trajetória ausente da tabela (contados como impossíveis).")
    out['table'] = names[impossible].value_counts().rename('n_pixels').reset_index().head(top) if n_imp else names.iloc[:0]
    return out
