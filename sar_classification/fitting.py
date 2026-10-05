"""
Ajuste de distribuições e testes de aderência (origem: script 12).

ks_test e fit_and_test funcionam para qualquer distribuição univariada do catálogo que
tenha 'cdf'. As funções gamma_* e normal_* são atalhos com os nomes e colunas do original.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .distributions import as_2d, check_data, get_distribution
from .samples import split_by_class

PARAM_LABELS = {'gamma': {'alpha': 'Alpha', 'beta': 'Beta'}}


# =============================================================================
# KS genérico
# =============================================================================
def ks_pvalue(x, cdf_values_fn):
    """
    KS de uma amostra com a regra do ks.test do R: p exato se n < 100 e sem empates;
    assintótico caso contrário.
    """
    x = np.asarray(x, float)
    exact = len(x) < 100 and len(np.unique(x)) == len(x)
    res = stats.kstest(x, cdf_values_fn, method='exact' if exact else 'asymp')
    return float(res.statistic), float(res.pvalue)


def ks_test(x, distribution, params, ENL=None):
    """KS de x contra a distribuição do catálogo com os parâmetros dados. Returns (D, p)."""
    dist = get_distribution(distribution)
    x = as_2d(x)
    if x.shape[1] != 1 or dist['cdf'] is None:
        raise ValueError(f"KS requer dado univariado e cdf disponível ('{distribution}').")
    return ks_pvalue(x[:, 0], lambda v: dist['cdf'](v, params, ENL))


def _flatten(distribution, params):
    labels = PARAM_LABELS.get(distribution, {})
    if distribution == 'gaussian':
        return {'Mean': float(params['mean'][0]), 'SD': float(np.sqrt(params['covariance_matrix'][0, 0]))}
    return {labels.get(k, k): float(v) for k, v in params.items()}


def fit_and_test(df_samples, classes, distribution, ENL=None, method=None, significance_level=0.05,
                 verbose=True):
    """
    Para cada classe: ajusta a distribuição e aplica o KS.
    Returns: DataFrame Class, <parâmetros>, D, p_value, H0 ('accept' se p > significance_level).
    """
    dist = get_distribution(distribution)
    method = method or dist['fit_methods'][0]
    rows = []
    for c, x in split_by_class(df_samples, classes).items():
        check_data(distribution, x)
        params = dist['fit'](x, ENL=ENL, method=method)
        d, p = ks_test(x, distribution, params, ENL)
        h0 = 'accept' if p > significance_level else 'reject'
        rows.append({'Class': c, **_flatten(distribution, params), 'D': d, 'p_value': p, 'H0': h0})
        if verbose:
            print(f"Classe {c}: " + ' | '.join(f"{k} = {v:.6g}" for k, v in rows[-1].items()
                                                if k not in ('Class', 'H0')) + f" -> H0 {h0}")
    return pd.DataFrame(rows)


def plot_fit(x, distribution, params, ENL=None, title=None, bins='auto', pdf=None):
    """Histograma + densidade ajustada (univariado)."""
    import matplotlib.pyplot as plt
    dist = get_distribution(distribution)
    x = as_2d(x)[:, 0]
    grid = np.linspace(x.min(), x.max(), 400)
    _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins=bins, density=True, color='aliceblue', edgecolor='black')
    ax.plot(grid, np.exp(dist['logpdf'](grid, params, ENL)), color='red', lw=2)
    d, p = ks_test(x, distribution, params, ENL)
    txt = [f"{k} = {v:.4g}" for k, v in _flatten(distribution, params).items()] + [f"p-valor KS = {p:.4f}"]
    ax.text(0.97, 0.95, '\n'.join(txt), transform=ax.transAxes, ha='right', va='top', color='darkblue')
    ax.set_title(title or dist['label'])
    ax.set_xlabel('Valores')
    ax.set_ylabel('Densidade')
    if pdf is not None:
        pdf.savefig(plt.gcf(), bbox_inches='tight')
    plt.show()


# =============================================================================
# Atalhos com os nomes do original
# =============================================================================
def gamma_fitting(df_samples, classes, significance_level=0.05, method='mle', verbose=True):
    """Gama com α e β livres + KS (gamma_fitting)."""
    return fit_and_test(df_samples, classes, 'gamma', None, method, significance_level, verbose)


def gamma_fitting_fixed_enl(df_samples, classes, enl, significance_level=0.05, verbose=False):
    """Gama com α = ENL fixo e β = ENL/média + KS (gamma_fitting_specific)."""
    return fit_and_test(df_samples, classes, 'gamma', enl, None, significance_level, verbose)


def normal_fitting(df_samples, classes, significance_level=0.05, verbose=True):
    """Normal univariada por MV (σ com denominador n) + KS."""
    return fit_and_test(df_samples, classes, 'gaussian', None, 'mle', significance_level, verbose)


def best_enl(fitting_result):
    """α da classe com maior p-valor entre as que aceitaram H0 (NaN se nenhuma)."""
    acc = fitting_result[fitting_result['H0'] == 'accept']
    return np.nan if acc.empty else float(acc.loc[acc['p_value'].idxmax(), 'Alpha'])


def iterative_gamma_fitting(df_samples, classes, significance_level=0.05, method='mle', verbose=True):
    """
    ENL que faz mais classes aderirem à Gama de forma fixa:
    1. ajuste livre; candidatos = α das classes aceitas, em ordem decrescente de p-valor;
    2. para cada candidato, Gama com α = candidato em todas as classes;
    3. vencedor = candidato com mais aceitações (empate: o primeiro da ordem do passo 1).
    Returns: dict best_enl, fitting_result, result_counting, free_fit — ou None.
    """
    free = gamma_fitting(df_samples, classes, significance_level, method, verbose)
    cand = free[(free['H0'] == 'accept') & free['p_value'].notna()]
    if cand.empty:
        print("Nenhuma classe aderiu à Gama no ajuste livre: ENL não determinado.")
        return None
    cand = cand.sort_values('p_value', ascending=False, kind='stable')
    fits = {enl: gamma_fitting_fixed_enl(df_samples, classes, enl, significance_level) for enl in cand['Alpha']}
    counts = pd.DataFrame([{'ENL': e, 'Count': int((r['H0'] == 'accept').sum())} for e, r in fits.items()])
    winner = float(counts.loc[counts['Count'].idxmax(), 'ENL'])
    if verbose:
        print(f"\nENL vencedor = {winner:.6g} ({counts['Count'].max()} de {len(classes)} classes aceitas)")
    return {'best_enl': winner, 'fitting_result': fits[winner], 'result_counting': counts, 'free_fit': free}


# =============================================================================
# Mardia (normalidade multivariada)
# =============================================================================
def mardia_test(x, significance_level=0.05):
    """
    Assimetria e curtose multivariadas (formulação do pacote MVN: covariância com
    denominador n; correção de pequenas amostras na assimetria se n < 20).
    H0 aceita se os dois p-valores > significance_level.
    """
    x = as_2d(x)
    n, p = x.shape
    xc = x - x.mean(axis=0)
    d = xc @ np.linalg.solve(np.atleast_2d(np.cov(xc, rowvar=False, ddof=0)), xc.T)
    g1p, g2p = np.sum(d ** 3) / n ** 2, np.sum(np.diag(d) ** 2) / n
    df = p * (p + 1) * (p + 2) / 6
    if n < 20:
        skew = n * (((p + 1) * (n + 1) * (n + 3)) / (n * ((n + 1) * (p + 1) - 6))) * g1p / 6
    else:
        skew = n * g1p / 6
    kurt = (g2p - p * (p + 2)) * np.sqrt(n / (8 * p * (p + 2)))
    p_skew, p_kurt = float(stats.chi2.sf(skew, df)), float(2 * stats.norm.sf(abs(kurt)))
    h0 = 'accept' if (p_skew > significance_level and p_kurt > significance_level) else 'reject'
    return {'skewness': float(skew), 'p_value_skewness': p_skew,
            'kurtosis': float(kurt), 'p_value_kurtosis': p_kurt, 'H0': h0}


def normal_fitting_mv(df_samples, classes, significance_level=0.05):
    """Teste de Mardia para cada classe."""
    return pd.DataFrame([{'Class': c, **mardia_test(x, significance_level)}
                         for c, x in split_by_class(df_samples, classes).items()])
