"""
Verificação automática de uma distribuição do catálogo.

Use check_distribution sempre que cadastrar uma distribuição nova: ela confere, sem
depender de nenhuma referência externa, se as funções registradas são coerentes entre si.
"""
import numpy as np
import pandas as pd
from scipy import integrate

from .distributions import as_2d, get_distribution


def _support(domain):
    return (0.0, np.inf) if domain in ('intensity', 'amplitude') else (-np.inf, np.inf)


def check_distribution(name, params, ENL=None, n=20000, rng=0, rtol_fit=0.05, method=None):
    """
    Testes de coerência (parâmetros 'params' são os verdadeiros, usados na simulação):

    1. Normalização   : ∫ f = 1 (integração numérica; univariada ou bivariada).
    2. Simulação × cdf: KS da amostra simulada contra a cdf (univariadas com cdf).
    3. Recuperação    : fit(amostra simulada) ≈ params (erro relativo < rtol_fit).
    4. Bhattacharyya  : forma fechada × -ln ∫ sqrt(f1 f2) numérica (univariadas),
                        entre params e uma versão perturbada.

    Returns: DataFrame (teste, resultado, detalhe).
    """
    dist = get_distribution(name)
    rows = []
    lo, hi = _support(dist['domain'])
    pdf = lambda *v: float(np.exp(dist['logpdf'](np.array([v]), params, ENL))[0])

    # 1. Normalização
    p = dist['n_features']
    if p == 1 or (p is None and np.size(params.get('mean', [0])) == 1):
        total = integrate.quad(pdf, lo, hi, limit=200)[0]
    elif p == 2 or (p is None and np.size(params.get('mean', [])) == 2):
        total = integrate.dblquad(lambda y, x: pdf(x, y), lo, hi, lo, hi, epsabs=1e-7)[0]
    else:
        total = np.nan
    rows.append(('Normalização ∫f = 1', 'OK' if abs(total - 1) < 1e-4 else ('—' if np.isnan(total) else 'FALHOU'),
                 f'{total:.6f}'))

    # 2. Simulação e 3. recuperação
    x = as_2d(dist['sample'](n, params, ENL, rng))
    if dist['cdf'] is not None and x.shape[1] == 1:
        from .fitting import ks_pvalue
        d, pv = ks_pvalue(x[:, 0], lambda v: dist['cdf'](v, params, ENL))
        rows.append(('Simulação × cdf (KS)', 'OK' if pv > 0.01 else 'FALHOU', f'D = {d:.4f}, p = {pv:.3f}'))
    est = dist['fit'](x, ENL=ENL, method=method or dist['fit_methods'][0])
    errs = {k: float(np.max(np.abs(np.asarray(est[k]) - np.asarray(v)) / np.maximum(np.abs(np.asarray(v)), 1e-12)))
            for k, v in params.items()}
    rows.append(('Recuperação dos parâmetros', 'OK' if max(errs.values()) < rtol_fit else 'FALHOU',
                 ', '.join(f'{k}: {e:.2%}' for k, e in errs.items())))

    # 4. Bhattacharyya
    if dist['bhattacharyya'] is not None and x.shape[1] == 1:
        p2 = {k: np.asarray(v) * (1.3 if k not in ('alpha',) else 1.0) for k, v in params.items()}
        p2 = {k: (float(v) if np.ndim(v) == 0 else v) for k, v in p2.items()}
        f1 = lambda v: np.exp(dist['logpdf'](np.array([[v]]), params, ENL))[0]
        f2 = lambda v: np.exp(dist['logpdf'](np.array([[v]]), p2, ENL))[0]
        num = -np.log(integrate.quad(lambda v: np.sqrt(f1(v) * f2(v)), lo, hi, limit=200)[0])
        closed = dist['bhattacharyya'](params, p2, ENL)
        rows.append(('Bhattacharyya fechada × numérica', 'OK' if np.isclose(num, closed, rtol=1e-5) else 'FALHOU',
                     f'{closed:.6f} × {num:.6f}'))
    return pd.DataFrame(rows, columns=['teste', 'resultado', 'detalhe'])
