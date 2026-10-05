"""
Conjunto amostral (origem: script 11).

Convenção: df_samples é um DataFrame com a coluna 'Class' e as demais colunas numéricas
(atributos). Nenhuma função depende da posição da coluna 'Class'.
"""
import math

import numpy as np
import pandas as pd

CLASS_COL = 'Class'


def feature_columns(df):
    """Colunas de atributos (todas exceto 'Class')."""
    return [c for c in df.columns if c != CLASS_COL]


def check_samples(df, classes=None):
    """Confere a existência da coluna 'Class' e das classes pedidas."""
    if CLASS_COL not in df.columns:
        raise ValueError(f"O DataFrame deve conter a coluna '{CLASS_COL}'.")
    if classes is not None:
        missing = set(classes) - set(df[CLASS_COL].unique())
        if missing:
            raise ValueError(f"Classes ausentes nas amostras: {sorted(missing)}")


def build_samples_table(attribute_list, classes):
    """Junta uma tabela/array de atributos por classe em df_samples (df_samples_creation)."""
    if len(attribute_list) != len(classes):
        raise ValueError("attribute_list e classes devem ter o mesmo tamanho.")
    parts = []
    for cls, attrs in zip(classes, attribute_list):
        part = pd.DataFrame(attrs).reset_index(drop=True)
        part.insert(0, CLASS_COL, cls)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def split_by_class(df_samples, classes):
    """{classe: array (n x p)} na ordem de `classes`."""
    check_samples(df_samples, classes)
    feats = feature_columns(df_samples)
    return {c: df_samples.loc[df_samples[CLASS_COL] == c, feats].to_numpy(dtype=np.float64)
            for c in classes}


def downsample_samples(df_samples, ratio=1.0, random_state=None):
    """Limita cada classe a floor(ratio x nº de amostras da menor classe)."""
    check_samples(df_samples)
    rng = np.random.default_rng(random_state)
    max_n = int(math.floor(df_samples[CLASS_COL].value_counts().min() * ratio))
    parts = []
    for _, group in df_samples.groupby(CLASS_COL, sort=True):
        if len(group) > max_n:
            group = group.iloc[rng.choice(len(group), size=max_n, replace=False)]
        parts.append(group)
    return pd.concat(parts)


def generalize_classes(df_samples, hierarchical_levels, level=1):
    """
    Funde as classes de um nível hierárquico em 'C1+C2+...' (level 1-based, como no R).
    hierarchical_levels: lista de níveis; cada nível é uma lista de grupos de classes.
    """
    check_samples(df_samples)
    out = df_samples.copy()
    groups = hierarchical_levels[level - 1]
    if groups and isinstance(groups[0], str):
        groups = [groups]
    for group in groups:
        out.loc[out[CLASS_COL].isin(group), CLASS_COL] = '+'.join(group)
    return out


def split_samples(df_samples, ratio=None, samples_per_class=None, random_state=None):
    """
    Divisão estratificada treino/validação.

    ratio            : treino = ceil(n x ratio) por classe (regra do caret::createDataPartition;
                       classe com 1 amostra vai inteira para o treino).
    samples_per_class: nº fixo de treino por classe; classes com n <= samples_per_class
                       vão inteiras para o treino (sem validação).
    """
    check_samples(df_samples)
    if ratio is None and samples_per_class is None:
        raise ValueError("Informe ratio ou samples_per_class.")
    rng = np.random.default_rng(random_state)
    train_idx = []
    for _, group in df_samples.groupby(CLASS_COL, sort=True):
        n = len(group)
        if samples_per_class is not None:
            k = n if n <= samples_per_class else samples_per_class
        else:
            k = n if n == 1 else int(math.ceil(n * ratio))
        train_idx.extend(rng.choice(group.index.to_numpy(), size=k, replace=False))
    mask = df_samples.index.isin(train_idx)
    return df_samples[mask], df_samples[~mask]


def drop_duplicates_by_class(df_samples, columns=None):
    """
    Remove, dentro de cada classe, as amostras em que QUALQUER atributo de `columns` repete um
    valor já visto naquela classe (mantém a 1ª ocorrência). Equivale ao filtro dos scripts R:
        group_by(Class) %>% filter(!duplicated(HH) & !duplicated(HV))
    """
    check_samples(df_samples)
    columns = columns or feature_columns(df_samples)
    keep = np.ones(len(df_samples), bool)
    for col in columns:
        keep &= ~df_samples.groupby(CLASS_COL, sort=False)[col].transform(lambda s: s.duplicated()).to_numpy(bool)
    out = df_samples[keep]
    removed = len(df_samples) - len(out)
    if removed:
        print(f"drop_duplicates_by_class: {removed} amostras com valores repetidos removidas.")
    return out


def describe_by_class(df_samples, columns=None):
    """
    Estatística descritiva por classe e atributo (Máximo, Mínimo, Média, Variância (n-1),
    Mediana, Contagem), em formato longo: uma linha por (atributo, classe).
    """
    check_samples(df_samples)
    rows = []
    for col in columns or feature_columns(df_samples):
        g = df_samples.groupby(CLASS_COL, sort=False)[col]
        t = pd.DataFrame({'Maximo': g.max(), 'Minimo': g.min(), 'Media': g.mean(), 'Variancia': g.var(ddof=1),
                          'Mediana': g.median(), 'Contagem': g.size()})
        t.insert(0, 'Atributo', col)
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)
