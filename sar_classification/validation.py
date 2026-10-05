"""
Matriz de confusão, métricas e validação cruzada (origem: script 13).

Estrutura da validação cruzada
------------------------------
make_splits      -> gera as divisões (semente, treino, validação)
evaluate_split   -> avalia UMA divisão: fit_model -> classify -> matriz -> agregação -> métricas
cross_validation -> percorre as divisões e organiza os resultados
select_seeds     -> sementes da melhor iteração, da mediana e do 3º quartil, pela métrica escolhida
"""
import warnings

import numpy as np
import pandas as pd

from .models import classify, fit_model
from .samples import CLASS_COL, split_samples


# =============================================================================
# Matriz de confusão
# =============================================================================
def match_classes(predicted, reference, rename_predicted=None):
    """
    DataFrame com 'predicted_class' e 'Class'. rename_predicted ({antigo: novo}) renomeia
    apenas a previsão (comportamento do original).
    """
    pred = pd.Series(np.asarray(predicted, dtype=object), name='predicted_class')
    ref = pd.Series(np.asarray(reference, dtype=object), name=CLASS_COL)
    if rename_predicted:
        pred = pred.replace(rename_predicted)
    return pd.concat([pred, ref], axis=1)


def confusion_matrix_table(df_temp, labels=None):
    """Linhas = Prediction, colunas = Reference; classes = união ordenada (padrão)."""
    pred, ref = df_temp['predicted_class'], df_temp[CLASS_COL]
    if labels is None:
        labels = sorted(set(pred.dropna()) | set(ref.dropna()))
    cm = pd.crosstab(pd.Categorical(pred, categories=labels), pd.Categorical(ref, categories=labels),
                     dropna=False).reindex(index=labels, columns=labels, fill_value=0)
    cm.index.name, cm.columns.name = 'Prediction', 'Reference'
    return cm


def aggregate_classes(confusion_matrix, classes_to_aggregate):
    """Soma linhas e colunas das classes em 'C1+C2+...', posicionada no final."""
    if len(classes_to_aggregate) < 2:
        raise ValueError("Informe ao menos duas classes para agregar.")
    missing = set(classes_to_aggregate) - set(confusion_matrix.index)
    if missing:
        raise ValueError(f"Classes ausentes na matriz de confusão: {sorted(missing)}")
    new = '+'.join(classes_to_aggregate)
    mapping = {c: new if c in classes_to_aggregate else c for c in confusion_matrix.index}
    order = [c for c in confusion_matrix.index if c not in classes_to_aggregate] + [new]
    out = (confusion_matrix.rename(index=mapping, columns=mapping)
           .T.groupby(level=0).sum().T.groupby(level=0).sum()).loc[order, order]
    out.index.name, out.columns.name = 'Prediction', 'Reference'
    return out


# =============================================================================
# Métricas
# =============================================================================
def metrics_calculation(confusion_matrix, normalize=True):
    """
    normalize=True (original): colunas divididas pelo total antes do cálculo ->
        overall_accuracy = média das PAs (acurácia balanceada); UA com referências equiprováveis.
    normalize=False: métricas convencionais.
    Classes sem referência são excluídas (com aviso); F1 = 0 quando PA = UA = 0.
    """
    cm = confusion_matrix.astype(float)
    col_tot = cm.sum(axis=0)
    empty = col_tot.index[col_tot == 0].tolist()
    if empty:
        warnings.warn(f"Classes sem amostras de referência (excluídas das métricas): {empty}")
        keep = [c for c in cm.columns if c not in empty]
        cm, col_tot = cm.loc[keep, keep], col_tot[keep]
    mat = cm / col_tot if normalize else cm
    diag = pd.Series(np.diag(mat), index=mat.columns)
    producers = (diag / mat.sum(axis=0)).fillna(0.0)
    users = (diag / mat.sum(axis=1).reindex(mat.columns)).fillna(0.0)
    denom = producers + users
    f1 = (2 * producers * users / denom).where(denom > 0, 0.0)
    return {'overall_accuracy': float(diag.sum() / mat.to_numpy().sum()), 'f1_macro': float(f1.mean()),
            'producers_accuracy': producers, 'users_accuracy': users, 'f1_score': f1}


# =============================================================================
# Validação cruzada
# =============================================================================
def make_splits(df_samples, seeds, train_proportion=0.7, mode='variable',
                external_valid_samples=None, n_samples=None):
    """
    Gera (semente, treino, validação) para cada semente.
    mode='variable': sorteia treino/validação em df_samples.
    mode='fixed'   : treino = 100% de df_samples; validação sorteada em external_valid_samples.
    Com external_valid_samples, a validação vem sempre dela
    (train_proportion ou n_samples por classe).
    """
    if mode not in ('variable', 'fixed'):
        raise ValueError("mode deve ser 'variable' ou 'fixed'.")
    if mode == 'fixed' and external_valid_samples is None:
        raise ValueError("mode='fixed' exige external_valid_samples (o treino usa 100% das amostras).")
    train_ratio = 1.0 if mode == 'fixed' else train_proportion
    for seed in seeds:
        rng = np.random.default_rng(seed)
        train, test = split_samples(df_samples, ratio=train_ratio, random_state=rng)
        if external_valid_samples is not None:
            test, _ = split_samples(external_valid_samples, ratio=train_proportion,
                                    samples_per_class=n_samples, random_state=rng)
        yield seed, train, test


def evaluate_split(train, test, classes, distribution='gaussian', ENL=None, aggregation=None,
                   rename_predicted=None, normalize=True, transform=None):
    """
    Avalia UMA divisão. Returns: dict com model, predicted, confusion_matrix e as métricas.
    """
    model = fit_model(train, classes, distribution, ENL, transform=transform)
    pred = classify(test, model)
    cm = confusion_matrix_table(match_classes(pred, test[CLASS_COL], rename_predicted))
    for group in aggregation or []:
        cm = aggregate_classes(cm, group)
    return {'model': model, 'predicted': pred, 'confusion_matrix': cm,
            **metrics_calculation(cm, normalize=normalize)}


def select_seeds(results, rank_by='f1_macro'):
    """
    Sementes da melhor iteração e das posições floor(n/2) (mediana) e ceil(0,75 n) (3º quartil)
    na ordenação crescente da métrica `rank_by` ('f1_macro' — como no script 13 —
    ou 'overall_accuracy' — como descrito na Seção 4.3.1.4 da dissertação).
    """
    values = np.asarray(results[rank_by], float)
    seeds = results['seeds']
    order = np.argsort(values, kind='stable')
    n = len(values)
    return {'rank_by': rank_by,
            'best_seed': seeds[int(np.nanargmax(values))],
            'median_seed': seeds[order[max(n // 2, 1) - 1]],
            'third_quartile_seed': seeds[order[int(np.ceil(0.75 * n)) - 1]]}


def cross_validation(df_samples, classes, distribution='gaussian', ENL=None, n_iterations=10,
                     seeds=None, train_proportion=0.7, mode='variable', external_valid_samples=None,
                     n_samples=None, aggregation=None, rename_predicted=None, normalize=True,
                     rank_by='f1_macro', keep_confusion_matrices=False, verbose=True, transform=None):
    """
    Holdout estratificado repetido com o classificador MaxVer.

    seeds: lista de sementes (uma por iteração); None -> 1..n_iterations.
    rank_by: métrica que define as sementes de melhor/mediana/3º quartil.

    Returns: dict com overall_accuracy, f1_macro (arrays), producers_accuracy, users_accuracy,
    f1_score (DataFrames semente x classe), seeds, selected (dict de select_seeds) e,
    se keep_confusion_matrices, confusion_matrices {semente: matriz}.
    """
    seeds = list(range(1, n_iterations + 1)) if seeds is None else list(seeds)
    rows = {'overall_accuracy': [], 'f1_macro': [], 'producers_accuracy': [],
            'users_accuracy': [], 'f1_score': []}
    cms = {}
    for it, (seed, train, test) in enumerate(make_splits(df_samples, seeds, train_proportion, mode,
                                                          external_valid_samples, n_samples), start=1):
        r = evaluate_split(train, test, classes, distribution, ENL, aggregation, rename_predicted, normalize, transform)
        for k in rows:
            rows[k].append(r[k])
        if keep_confusion_matrices:
            cms[seed] = r['confusion_matrix']
        if verbose:
            print(f"Iteração {it}/{len(seeds)} (seed {seed}): OA = {r['overall_accuracy']:.4f} | "
                  f"F1 macro = {r['f1_macro']:.4f}")

    idx = pd.Index(seeds, name='seed')
    results = {'overall_accuracy': np.asarray(rows['overall_accuracy']),
               'f1_macro': np.asarray(rows['f1_macro']),
               'producers_accuracy': pd.DataFrame(rows['producers_accuracy'], index=idx),
               'users_accuracy': pd.DataFrame(rows['users_accuracy'], index=idx),
               'f1_score': pd.DataFrame(rows['f1_score'], index=idx),
               'seeds': seeds}
    results['selected'] = select_seeds(results, rank_by)
    if keep_confusion_matrices:
        results['confusion_matrices'] = cms
    return results


def reproduce_iteration(df_samples, classes, seed, distribution='gaussian', ENL=None, **kwargs):
    """
    Refaz exatamente a iteração de uma semente (ex.: a do 3º quartil), devolvendo o modelo,
    a previsão e a matriz de confusão. kwargs: os mesmos de cross_validation
    (train_proportion, mode, external_valid_samples, n_samples, aggregation, rename_predicted, normalize).
    """
    split_keys = ('train_proportion', 'mode', 'external_valid_samples', 'n_samples')
    eval_keys = ('aggregation', 'rename_predicted', 'normalize', 'transform')
    _, train, test = next(make_splits(df_samples, [seed], **{k: kwargs[k] for k in split_keys if k in kwargs}))
    out = evaluate_split(train, test, classes, distribution, ENL, **{k: kwargs[k] for k in eval_keys if k in kwargs})
    out.update({'train': train, 'test': test, 'seed': seed})
    return out


# =============================================================================
# Estatísticas
# =============================================================================
def calc_stats(df):
    """Média, DP (n-1) e percentis 2,5% / 97,5% (intervalo empírico, não IC da média)."""
    df = pd.DataFrame(df)
    return pd.DataFrame({'Mean': df.mean(), 'SD': df.std(ddof=1),
                         'ci_lw': df.quantile(0.025), 'ci_up': df.quantile(0.975)})


def calc_stats_vector(vec):
    s = pd.Series(vec, dtype=float)
    return pd.DataFrame({'Mean': [s.mean()], 'SD': [s.std(ddof=1)],
                         'ci_lw': [s.quantile(0.025)], 'ci_up': [s.quantile(0.975)]})


def cross_validation_stats(cv_results):
    per_class = pd.concat([calc_stats(cv_results['producers_accuracy']).add_suffix('_PA'),
                           calc_stats(cv_results['users_accuracy']).add_suffix('_UA'),
                           calc_stats(cv_results['f1_score']).add_suffix('_F1')], axis=1)
    return {'Overall_Stats': calc_stats_vector(cv_results['overall_accuracy']),
            'F1_Macro_Stats': calc_stats_vector(cv_results['f1_macro']),
            'Stats_per_class': per_class}


# =============================================================================
# Avaliação de mapas classificados (MaxVer ou CMAP) com amostras de validação
# =============================================================================
def map_accuracy_resampling(predicted, reference, n_iterations=1000, proportion=0.7, replace=True,
                            seeds=None, aggregation=None, rename=None, normalize=True, verbose=False):
    """
    Métricas de um mapa classificado, reamostrando as amostras de validação a cada iteração
    (Seção 4.3.2 da dissertação: "1000 métricas por classificação, variando apenas as amostras
    do conjunto de validação na proporção de 70% de forma estratificada e com reposição").

    predicted, reference: classes no mapa e de referência nos mesmos pixels (arrays de nomes).
    Por classe de referência, sorteia ceil(proportion x n) amostras (com ou sem reposição).
    aggregation: grupos de classes somados na matriz de confusão (ex.: nível lvl_2).
    rename: {nome: novo nome} aplicado aos rótulos da matriz DEPOIS da agregação
            (ex.: {'FP+FD': 'F'}, legenda final); rótulos que coincidirem são somados.

    Returns: dict no mesmo formato de cross_validation (aceito por cross_validation_stats).
    """
    pred = pd.Series(np.asarray(predicted, dtype=object))
    ref = pd.Series(np.asarray(reference, dtype=object))
    ok = pred.notna() & ref.notna()
    pred, ref = pred[ok].reset_index(drop=True), ref[ok].reset_index(drop=True)
    seeds = list(range(1, n_iterations + 1)) if seeds is None else list(seeds)
    groups = {c: np.flatnonzero(ref.to_numpy() == c) for c in sorted(ref.unique())}
    rows = {'overall_accuracy': [], 'f1_macro': [], 'producers_accuracy': [], 'users_accuracy': [], 'f1_score': []}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = np.concatenate([rng.choice(ix, size=int(np.ceil(proportion * len(ix))), replace=replace)
                              for ix in groups.values()])
        cm = confusion_matrix_table(pd.DataFrame({'predicted_class': pred.iloc[idx].to_numpy(),
                                                  CLASS_COL: ref.iloc[idx].to_numpy()}))
        for group in aggregation or []:
            if set(group) <= set(cm.index):
                cm = aggregate_classes(cm, group)
        if rename:
            cm = cm.rename(index=rename, columns=rename)
            cm = cm.T.groupby(level=0, sort=False).sum().T.groupby(level=0, sort=False).sum()
            cm.index.name, cm.columns.name = 'Prediction', 'Reference'
        m = metrics_calculation(cm, normalize=normalize)
        for k in rows:
            rows[k].append(m[k])
    ix = pd.Index(seeds, name='seed')
    out = {'overall_accuracy': np.asarray(rows['overall_accuracy']), 'f1_macro': np.asarray(rows['f1_macro']),
           'producers_accuracy': pd.DataFrame(rows['producers_accuracy'], index=ix),
           'users_accuracy': pd.DataFrame(rows['users_accuracy'], index=ix),
           'f1_score': pd.DataFrame(rows['f1_score'], index=ix), 'seeds': seeds}
    if verbose:
        print(f"OA média = {out['overall_accuracy'].mean():.4f} | F1 macro médio = {out['f1_macro'].mean():.4f} "
              f"({len(seeds)} reamostragens, {len(ref)} amostras de validação)")
    return out


def summarize_models(results, label='Modelo'):
    """
    Junta as estatísticas de vários resultados ({nome: saída de cross_validation ou
    map_accuracy_resampling}) em duas tabelas longas:
        global  : uma linha por modelo (OA e F1 macro: média, DP, ci_lw, ci_up)
        classes : uma linha por (modelo, classe) com PA, UA e F1
    """
    glob, per = [], []
    for name, r in results.items():
        st = cross_validation_stats(r)
        o, f = st['Overall_Stats'].iloc[0], st['F1_Macro_Stats'].iloc[0]
        glob.append({label: name, 'OA_media': o['Mean'], 'OA_dp': o['SD'], 'OA_ci_lw': o['ci_lw'], 'OA_ci_up': o['ci_up'],
                     'F1_media': f['Mean'], 'F1_dp': f['SD'], 'F1_ci_lw': f['ci_lw'], 'F1_ci_up': f['ci_up'],
                     'n_iteracoes': len(r['overall_accuracy'])})
        pc = st['Stats_per_class'].reset_index().rename(columns={'index': 'Classe', 'Reference': 'Classe'})
        pc.insert(0, label, name)
        per.append(pc)
    return pd.DataFrame(glob), pd.concat(per, ignore_index=True)


def iterations_table(results, label='Modelo'):
    """Métricas de cada iteração, de todos os modelos, numa só tabela longa (substitui os *_OAlist.csv)."""
    return pd.concat([pd.DataFrame({label: name, 'seed': r['seeds'], 'OA': r['overall_accuracy'], 'F1_macro': r['f1_macro']})
                      for name, r in results.items()], ignore_index=True)
