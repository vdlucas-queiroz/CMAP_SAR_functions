"""
Modelo de classes e classificador MaxVer (origem: script 11).

Um "modelo" é um dicionário autocontido:
    {'distribution': nome no catálogo, 'ENL': float | None, 'features': [atributos do treino],
     'classes': [ordem das classes], 'params': {classe: parâmetros}}

Distribuição, ENL e atributos ficam gravados no modelo: a verossimilhança é sempre calculada
com a mesma distribuição, o mesmo ENL e os mesmos atributos (na mesma ordem) da estimação.
"""
import json

import numpy as np
import pandas as pd

from .distributions import as_2d, check_data, get_distribution
from .samples import feature_columns, split_by_class

# Transformações aplicadas aos atributos ANTES da estimação e da verossimilhança.
# Ficam gravadas no modelo, de modo que treino e aplicação usam sempre a mesma.
TRANSFORMS = {
    None: lambda x: x,
    'sqrt': np.sqrt,                                   # intensidade -> amplitude
    'db': lambda x: 10.0 * np.log10(x),                # intensidade -> dB
}


def apply_transform(x, transform):
    """Aplica a transformação registrada (None, 'sqrt' ou 'db')."""
    if transform not in TRANSFORMS:
        raise ValueError(f"Transformação '{transform}' desconhecida. Disponíveis: {list(TRANSFORMS)}")
    with np.errstate(invalid='ignore', divide='ignore'):
        return TRANSFORMS[transform](x)


def fit_model(df_samples, classes, distribution='gaussian', ENL=None, method=None, transform=None):
    """
    Estima os parâmetros de cada classe (equivalente a parameters_estimation).

    method: estimador do catálogo; None usa o padrão da distribuição
            (Gaussiana: covariância n-1; Gama: α = ENL e β = ENL/média; par: momentos).
    transform: None, 'sqrt' (amplitude) ou 'db', aplicada aos atributos antes da estimação;
               fica gravada no modelo e é reaplicada automaticamente em class_likelihood.
    """
    dist = get_distribution(distribution)
    if dist['requires_enl'] and ENL is None:
        raise ValueError(f"A distribuição '{distribution}' exige o ENL.")
    method = method or dist['fit_methods'][0]
    params = {}
    for c, x in split_by_class(df_samples, classes).items():
        x = apply_transform(x, transform)
        check_data(distribution, x)
        params[c] = dist['fit'](x, ENL=ENL, method=method)
    return {'distribution': distribution, 'ENL': None if ENL is None else float(ENL),
            'features': feature_columns(df_samples), 'transform': transform,
            'classes': list(classes), 'params': params}


def _model_matrix(data, model):
    """
    Matriz de atributos na ordem do treino. DataFrame: colunas selecionadas e reordenadas
    pelo nome; array: exige o mesmo nº de atributos.
    """
    feats = model['features']
    if isinstance(data, pd.DataFrame):
        missing = [f for f in feats if f not in data.columns]
        if missing:
            raise ValueError(f"Atributos do modelo ausentes nos dados: {missing} (modelo: {feats}).")
        return as_2d(data[feats].to_numpy())
    x = as_2d(data)
    if x.shape[1] != len(feats):
        raise ValueError(f"Os dados têm {x.shape[1]} atributo(s); o modelo foi treinado com {len(feats)}: {feats}.")
    return x


def class_likelihood(data, model, log=False):
    """
    (Log-)verossimilhança de cada amostra para cada classe do modelo.

    data: DataFrame (atributos localizados pelo nome; colunas extras são ignoradas) ou
          array n x p na ordem de model['features'].
    Returns: DataFrame n x classes (colunas = nomes das classes).
    """
    dist = get_distribution(model['distribution'])
    x = apply_transform(_model_matrix(data, model), model.get('transform'))
    logp = np.column_stack([dist['logpdf'](x, model['params'][c], model['ENL']) for c in model['classes']])
    index = data.index if isinstance(data, pd.DataFrame) else None
    out = pd.DataFrame(logp, columns=model['classes'], index=index)
    return out if log else np.exp(out)


def maximum_likelihood_classifier(likelihoods):
    """
    Classe de maior (log-)verossimilhança por amostra (Eq. 2.5). Empate: primeira coluna
    (max.col ties.method='first'). Amostras com todas as colunas NaN -> NaN.
    """
    values = likelihoods.to_numpy(float)
    all_nan = np.all(np.isnan(values), axis=1)
    idx = np.argmax(np.where(np.isnan(values), -np.inf, values), axis=1)
    pred = pd.Series(np.asarray(likelihoods.columns, dtype=object)[idx], index=likelihoods.index,
                     name='predicted_class', dtype=object)
    pred[all_nan] = np.nan
    return pred


def classify(data, model):
    """Atalho: log-verossimilhança + MaxVer."""
    return maximum_likelihood_classifier(class_likelihood(data, model, log=True))


def save_model(model, path):
    """Grava o modelo em JSON (arrays convertidos em listas)."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(w) for k, w in v.items()}
        return v
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(conv(model), f, ensure_ascii=False, indent=2)
    return str(path)


def _restore(model):
    for c, p in model['params'].items():
        model['params'][c] = {k: np.asarray(v) if isinstance(v, list) else v for k, v in p.items()}
    model.setdefault('transform', None)
    return model


def load_model(path):
    """Lê um modelo gravado por save_model (listas numéricas voltam a ser arrays)."""
    with open(path, encoding='utf-8') as f:
        return _restore(json.load(f))


def save_models(models, path, extra=None):
    """Vários modelos ({nome: modelo}) num só JSON, com metadados opcionais (ex.: ENL, sementes)."""
    from .reporting import save_json
    return save_json({'models': models, 'extra': extra or {}}, path)


def load_models(path):
    """Lê o JSON de save_models. Returns: ({nome: modelo}, extra)."""
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return {k: _restore(v) for k, v in d['models'].items()}, d.get('extra', {})
