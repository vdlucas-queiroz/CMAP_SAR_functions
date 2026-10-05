"""
Separabilidade entre classes e agrupamento hierárquico (origem: script 11).

A distância de Bhattacharyya é calculada a partir dos parâmetros ajustados de cada classe,
com a forma fechada registrada no catálogo de distribuições.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

from .distributions import get_distribution
from .models import fit_model
from .samples import CLASS_COL, check_samples, feature_columns


def bhattacharyya_matrix(df_samples, classes, distribution='gaussian', ENL=None):
    """Matriz simétrica (DataFrame) de distâncias de Bhattacharyya."""
    check_samples(df_samples, classes)
    dist = get_distribution(distribution)
    if dist['bhattacharyya'] is None:
        raise NotImplementedError(f"Bhattacharyya em forma fechada não disponível para '{distribution}'.")
    if not feature_columns(df_samples):
        return None
    model = fit_model(df_samples, classes, distribution, ENL)
    n = len(classes)
    mat = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        mat[i, j] = mat[j, i] = dist['bhattacharyya'](model['params'][classes[i]],
                                                      model['params'][classes[j]], ENL)
    return pd.DataFrame(mat, index=classes, columns=classes)


def jeffries_matusita_matrix(df_samples, classes, distribution='gaussian', ENL=None):
    """JM = sqrt( 2 (1 - e^(-B)) ), no intervalo [0, √2]."""
    b = bhattacharyya_matrix(df_samples, classes, distribution, ENL)
    return None if b is None else np.sqrt(2.0 * (1.0 - np.exp(-b)))


def mean_jm_distance(df_samples, classes, distribution='gaussian', ENL=None):
    """Média das distâncias JM entre todos os pares de classes."""
    jm = jeffries_matusita_matrix(df_samples, classes, distribution, ENL)
    return np.nan if jm is None else float(np.nanmean(squareform(jm.to_numpy(), checks=False)))


def forward_selection_jm(df_samples, classes, all_features, distribution='gaussian', ENL=None):
    """
    Seleção de atributos pela média JM (exhaustive_search_jm_optimal): busca exaustiva no
    1º nível; depois, combinações de k atributos que contêm o melhor conjunto anterior.
    O conjunto-base só muda quando a média JM aumenta.

    Returns: (distances, results)
    """
    distances, results = [], []
    previous_max, base = 0.0, None
    for k in range(1, len(all_features) + 1):
        if base is not None:
            rest = [f for f in all_features if f not in base]
            combos = [list(base) + list(c) for c in itertools.combinations(rest, k - len(base))]
        else:
            combos = [list(c) for c in itertools.combinations(all_features, k)]
        best_dist, best_set = -np.inf, None
        for fs in combos:
            dist = mean_jm_distance(df_samples[[CLASS_COL] + fs], classes, distribution, ENL)
            distances.append({'Features': ', '.join(fs), 'JM_dist': dist})
            if not np.isnan(dist) and dist > best_dist:
                best_dist, best_set = dist, fs
        gain = abs(best_dist - previous_max) / previous_max if previous_max > 0 else 0.0
        results.append({'Features': ', '.join(best_set) if best_set else '', 'Gain': gain, 'JM_dist': best_dist})
        if k == 1 or best_dist > previous_max:
            base = best_set
        previous_max = best_dist
    return pd.DataFrame(distances), pd.DataFrame(results)


def hierarchical_clustering(dist_matrix, method='ward'):
    """Matriz de ligação do SciPy. method='ward' equivale a hclust(method='ward.D2') do R."""
    return hierarchy.linkage(squareform(dist_matrix.to_numpy(), checks=False), method=method)


def extract_hierarchy(linkage_matrix, labels):
    """{'lvl_1': [classes do 1º agrupamento], 'lvl_2': [...], ...}."""
    n = len(labels)
    members = {i: [labels[i]] for i in range(n)}
    levels = {}
    for step, (a, b) in enumerate(linkage_matrix[:, :2].astype(int)):
        members[n + step] = members[a] + members[b]
        levels[f'lvl_{step + 1}'] = members[n + step]
    return levels
