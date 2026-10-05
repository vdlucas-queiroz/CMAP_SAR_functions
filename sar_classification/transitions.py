"""
Matrizes de transição e probabilidade a priori das trajetórias (Eq. 2.12).
Usado pelo CMAP (cmap.py) e pela avaliação de trajetórias (trajectory_assessment.py).
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


def read_transition_matrix(path, sep=';', decimal='.', has_labels=True, class_names=None, tol=1e-6,
                           verbose=True):
    """
    Lê uma matriz de transição (linhas = classe em t-1, colunas = classe em t).

    has_labels=True : primeira linha e primeira coluna com os nomes das classes (formato do CMAP).
    has_labels=False: só números (formato de trajectory_assessment.R). Nomes das classes:
                      class_names=None -> '1'..'K' (linhas e colunas numeradas separadamente);
                      class_names=lista -> mesma lista para linhas e colunas;
                      class_names=(linhas, colunas) -> listas distintas.
    Linhas que não somam 1 geram aviso (esperado na matriz de validade).
    """
    if has_labels:
        df = pd.read_csv(path, sep=sep, index_col=0, decimal=decimal)
        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()
    else:
        df = pd.read_csv(path, sep=sep, header=None, decimal=decimal)
        if class_names is None:
            rows, cols = range(1, df.shape[0] + 1), range(1, df.shape[1] + 1)
        elif isinstance(class_names, tuple):
            rows, cols = class_names
        else:
            rows = cols = class_names
        if (len(rows), len(cols)) != df.shape:
            raise ValueError(f"'{path}': matriz {df.shape} incompatível com {len(rows)} x {len(cols)} nomes.")
        df.index, df.columns = [str(c) for c in rows], [str(c) for c in cols]
    if df.index.duplicated().any() or df.columns.duplicated().any():
        raise ValueError(f"Classes duplicadas na matriz '{path}'.")
    try:
        df = df.astype(float)
    except ValueError as err:
        raise ValueError(f"Valores não numéricos em '{path}' (verifique sep/decimal/has_labels): {err}")
    if (df.values < 0).any():
        raise ValueError(f"A matriz '{path}' contém valores negativos.")
    sums = df.sum(axis=1)
    off = sums[(sums - 1).abs() > tol]
    if len(off) and verbose:
        print(f"Aviso: em '{Path(path).name}', {len(off)} linha(s) não somam 1 (Eq. 2.13) — "
              f"esperado apenas em matriz de validade.")
    return df


def check_chain(matrices):
    """Confere se as colunas da matriz i são as linhas da matriz i+1."""
    for i in range(len(matrices) - 1):
        a, b = set(matrices[i].columns), set(matrices[i + 1].index)
        if a != b:
            raise ValueError(f"Stage_{i + 2}: destino da matriz {i + 1} {sorted(a)} ≠ origem da matriz {i + 2} {sorted(b)}.")


def trajectory_priors(matrices, initial_probs=None):
    """
    Todas as trajetórias (produto cartesiano das classes) e P(s) (Eq. 2.12).
    matrices: lista de DataFrames (read_transition_matrix). initial_probs: {classe: P(ω1)} ou None (uniforme).
    Returns: DataFrame Stage_1..Stage_T, weight_*, final_weight.
    """
    if not matrices:
        raise ValueError("Informe ao menos uma matriz de transição.")
    check_chain(matrices)
    stages = [list(matrices[0].index)] + [list(m.columns) for m in matrices]
    cols = [f'Stage_{i + 1}' for i in range(len(stages))]
    df = pd.DataFrame(list(itertools.product(*stages)), columns=cols)
    wcols = []
    if initial_probs is not None:
        init = {str(k).strip(): float(v) for k, v in initial_probs.items()}
        miss = set(stages[0]) - set(init)
        if miss:
            raise ValueError(f"initial_probs não contém as classes: {sorted(miss)}")
        df['weight_0'] = df['Stage_1'].map(init)
        wcols.append('weight_0')
    for i, m in enumerate(matrices):
        r = m.index.get_indexer(df[f'Stage_{i + 1}'])
        c = m.columns.get_indexer(df[f'Stage_{i + 2}'])
        df[f'weight_{i + 1}'] = m.to_numpy()[r, c]
        wcols.append(f'weight_{i + 1}')
    df['final_weight'] = df[wcols].prod(axis=1)
    return df


def create_transition_matrix_table(file_paths, output_path=None, sep=';', decimal='.', initial_probs=None,
                                   has_labels=True, class_names=None, verbose=True):
    """Lê as matrizes, calcula P(s) e (opcionalmente) grava o CSV. Mesma função do CMAP_FUNCTIONS."""
    mats = [read_transition_matrix(p, sep, decimal, has_labels, class_names, verbose=verbose) for p in file_paths]
    df = trajectory_priors(mats, initial_probs)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, sep=';', index=False)
    if verbose:
        n_valid = int((df['final_weight'] > 0).sum())
        print(f"Trajetórias: {len(df)} | P(s) > 0: {n_valid} | inválidas: {len(df) - n_valid}"
              + ("" if initial_probs is not None else " | P(ω1) uniforme"))
    return df


def read_trajectory_table(path, bands_config, sep=';', weight_column='final_weights'):
    """
    Lê uma tabela de trajetórias PRONTA no formato do CMAP_classifier do R (ex.: pesos_C4_C5.txt):
    uma coluna por data com o CÓDIGO da classe (posição da banda, 1-based) e a coluna de pesos.
    Converte os códigos em nomes de classe com bands_config (classes das bandas de cada data).
    Returns: DataFrame Stage_1..Stage_T, final_weight (formato de trajectory_priors).
    """
    raw = pd.read_csv(path, sep=sep)
    if weight_column not in raw.columns:
        raise ValueError(f"Coluna de pesos '{weight_column}' não encontrada em '{path}' (colunas: {list(raw.columns)}).")
    date_cols = [c for c in raw.columns if c != weight_column][:len(bands_config)]
    if len(date_cols) != len(bands_config):
        raise ValueError(f"'{path}' tem {len(date_cols)} colunas de data; bands_config tem {len(bands_config)}.")
    out = pd.DataFrame()
    for t, (col, cfg) in enumerate(zip(date_cols, bands_config)):
        codes = raw[col].astype(int)
        if codes.min() < 1 or codes.max() > len(cfg):
            raise ValueError(f"Coluna '{col}': códigos fora de 1..{len(cfg)}.")
        out[f'Stage_{t + 1}'] = [cfg[k - 1] for k in codes]
    out['final_weight'] = raw[weight_column].astype(float).to_numpy()
    return out
