"""
Catálogo de distribuições.

Cada distribuição é um dicionário com a MESMA interface. Todos os demais módulos
(estimação, verossimilhança, MaxVer, validação, testes de aderência, rasters) acessam
as distribuições apenas por este catálogo. Para incluir uma distribuição nova, basta
escrever suas funções e registrá-la em DISTRIBUTIONS — nada mais precisa ser alterado.

Interface de cada entrada
-------------------------
label          : str   — nome legível
domain         : str   — 'intensity', 'amplitude', 'any' ...
n_features     : int | None — nº de atributos exigido (None = qualquer nº >= 1)
requires_enl   : bool  — se o ENL é obrigatório na estimação
fit_methods    : tuple — métodos aceitos por `fit`
fit(x, ENL, method) -> dict de parâmetros
logpdf(x, params, ENL) -> array (n,)          — log-densidade vetorizada
cdf(x, params, ENL) -> array (n,) | None       — para o teste KS (univariadas)
sample(n, params, ENL, rng) -> array (n, p)    — simulação (testes)
bhattacharyya(params1, params2, ENL) -> float | None  — forma fechada, se existir

Convenção: x é sempre um array 2D (n amostras x p atributos), em float64.
"""
import numpy as np
from scipy import special, stats


# =============================================================================
# Utilidades
# =============================================================================
def as_2d(x):
    """Converte para array float64 2D (n x p)."""
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, 1) if x.ndim == 1 else x


def _require_enl(ENL, name):
    if ENL is None:
        raise ValueError(f"A distribuição '{name}' exige o ENL.")
    if ENL <= 0:
        raise ValueError(f"ENL deve ser positivo (recebido {ENL}).")
    return float(ENL)


# =============================================================================
# Gaussiana multivariada
# =============================================================================
def _gaussian_fit(x, ENL=None, method='unbiased'):
    """
    method='unbiased': covariância com denominador n-1 (cov() do R) — usado na classificação.
    method='mle'     : denominador n (máxima verossimilhança) — usado no teste KS,
                       como no fitdistrplus.
    """
    x = as_2d(x)
    ddof = {'unbiased': 1, 'mle': 0}[method]
    return {'mean': x.mean(axis=0), 'covariance_matrix': np.atleast_2d(np.cov(x, rowvar=False, ddof=ddof))}


def _gaussian_logpdf(x, params, ENL=None):
    x = as_2d(x)
    chol = np.linalg.cholesky(np.atleast_2d(params['covariance_matrix']))
    z = np.linalg.solve(chol, (x - params['mean']).T)
    maha = np.sum(z ** 2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (x.shape[1] * np.log(2 * np.pi) + logdet + maha)


def _gaussian_cdf(x, params, ENL=None):
    x = as_2d(x)
    if x.shape[1] != 1:
        return None
    sd = float(np.sqrt(np.atleast_2d(params['covariance_matrix'])[0, 0]))
    return stats.norm.cdf(x[:, 0], loc=float(params['mean'][0]), scale=sd)


def _gaussian_sample(n, params, ENL=None, rng=None):
    rng = np.random.default_rng(rng)
    return as_2d(rng.multivariate_normal(params['mean'], np.atleast_2d(params['covariance_matrix']), n))


def _gaussian_bhattacharyya(p1, p2, ENL=None):
    """B = 1/8 Δᵀ Σ⁻¹ Δ + 1/2 ln( |Σ| / sqrt(|Σ1||Σ2|) ),  Σ = (Σ1+Σ2)/2."""
    s1, s2 = np.atleast_2d(p1['covariance_matrix']), np.atleast_2d(p2['covariance_matrix'])
    s = (s1 + s2) / 2
    d = np.asarray(p1['mean']) - np.asarray(p2['mean'])
    ld, ld1, ld2 = (np.linalg.slogdet(m)[1] for m in (s, s1, s2))
    return float(0.125 * d @ np.linalg.solve(s, d) + 0.5 * (ld - 0.5 * (ld1 + ld2)))


# =============================================================================
# Gama (intensidade, univariada)
# =============================================================================
def _gamma_fit(x, ENL=None, method='mle'):
    """
    Com ENL : forma fixa α = ENL, taxa β = ENL / média (modelo usado na classificação).
    Sem ENL : α e β livres — method='mle' (máxima verossimilhança) ou
              'mme' (momentos, variância com denominador n, como no fitdistrplus).
    """
    x = as_2d(x)[:, 0]
    if ENL is not None:
        enl = _require_enl(ENL, 'gamma')
        return {'alpha': enl, 'beta': enl / x.mean()}
    if method == 'mle':
        alpha, _, scale = stats.gamma.fit(x, floc=0)
        return {'alpha': float(alpha), 'beta': float(1.0 / scale)}
    if method == 'mme':
        m, v = x.mean(), x.var(ddof=0)
        return {'alpha': float(m ** 2 / v), 'beta': float(m / v)}
    raise ValueError("Para a Gama livre, method deve ser 'mle' ou 'mme'.")


def _gamma_logpdf(x, params, ENL=None):
    return stats.gamma.logpdf(as_2d(x)[:, 0], a=params['alpha'], scale=1.0 / params['beta'])


def _gamma_cdf(x, params, ENL=None):
    return stats.gamma.cdf(as_2d(x)[:, 0], a=params['alpha'], scale=1.0 / params['beta'])


def _gamma_sample(n, params, ENL=None, rng=None):
    rng = np.random.default_rng(rng)
    return as_2d(rng.gamma(params['alpha'], 1.0 / params['beta'], n))


def _gamma_bhattacharyya(p1, p2, ENL=None):
    """
    Gamas com a mesma forma L e médias λ1, λ2:  B = L ln( (λ1+λ2) / (2 sqrt(λ1 λ2)) ).
    A forma usada é o ENL informado ou, na falta dele, o α comum aos dois modelos.
    """
    if ENL is None:
        if not np.isclose(p1['alpha'], p2['alpha']):
            raise ValueError("Bhattacharyya Gama em forma fechada exige a mesma forma α nas duas classes.")
        ENL = p1['alpha']
    l1, l2 = p1['alpha'] / p1['beta'], p2['alpha'] / p2['beta']
    return float(ENL * (np.log(l1 + l2) - np.log(2.0) - 0.5 * (np.log(l1) + np.log(l2))))


# =============================================================================
# Par de intensidades (bivariada)
# =============================================================================
def _pair_fit(x, ENL=None, method='moments'):
    """
    μ1, μ2 = médias; ρ² = |coef. de correlação entre as intensidades|, estimador de |ρc|².
    O ENL não é estimado aqui: entra na densidade.
    """
    x = as_2d(x)
    s1, s2 = x[:, 0], x[:, 1]
    mu1, mu2 = s1.mean(), s2.mean()
    e1 = np.mean((s1 - mu1) * (s2 - mu2))
    e2, e3 = np.mean((s1 - mu1) ** 2), np.mean((s2 - mu2) ** 2)
    return {'mu1': float(mu1), 'mu2': float(mu2), 'ro2': float(abs(e1 / np.sqrt(e2 * e3)))}


def _pair_logpdf(x, params, ENL=None):
    """
    Densidade conjunta de duas intensidades com n = ENL looks (Lee et al., 1994; Correia, 1998),
    em log, com Iν(z) = ive(ν, z)·e^z. Para ρ² = 0, usa o limite exato: produto de duas Gamas.
    """
    n = _require_enl(ENL, 'intensity_joint_distribution')
    x = as_2d(x)
    i1, i2 = x[:, 0], x[:, 1]
    mu1, mu2, r2 = params['mu1'], params['mu2'], params['ro2']
    with np.errstate(divide='ignore', invalid='ignore'):
        if r2 <= 0:
            return (stats.gamma.logpdf(i1, a=n, scale=mu1 / n) + stats.gamma.logpdf(i2, a=n, scale=mu2 / n))
        r = np.sqrt(r2)
        arg = (2 * n * r / (1 - r2)) * np.sqrt(i1 * i2 / (mu1 * mu2))
        return ((n + 1) * np.log(n) + 0.5 * (n - 1) * np.log(i1 * i2)
                - n * (i1 / mu1 + i2 / mu2) / (1 - r2)
                + np.log(special.ive(n - 1, arg)) + arg
                - 0.5 * (n + 1) * np.log(mu1 * mu2) - special.gammaln(n)
                - np.log(1 - r2) - (n - 1) * np.log(r))


def _pair_sample(n, params, ENL=None, rng=None):
    """Simula pelo modelo de speckle: média de L amostras complexas gaussianas correlacionadas."""
    L = _require_enl(ENL, 'intensity_joint_distribution')
    if not float(L).is_integer():
        raise ValueError("A simulação do par de intensidades exige ENL inteiro.")
    rng = np.random.default_rng(rng)
    L = int(L)
    rho = np.sqrt(params['ro2'])
    z1 = (rng.normal(size=(n, L)) + 1j * rng.normal(size=(n, L))) / np.sqrt(2)
    w = (rng.normal(size=(n, L)) + 1j * rng.normal(size=(n, L))) / np.sqrt(2)
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * w
    return np.column_stack([params['mu1'] * np.mean(abs(z1) ** 2, 1),
                            params['mu2'] * np.mean(abs(z2) ** 2, 1)])


# =============================================================================
# Catálogo
# =============================================================================
DISTRIBUTIONS = {
    'gaussian': {
        'label': 'Gaussiana multivariada', 'domain': 'any', 'n_features': None,
        'requires_enl': False, 'fit_methods': ('unbiased', 'mle'),
        'fit': _gaussian_fit, 'logpdf': _gaussian_logpdf, 'cdf': _gaussian_cdf,
        'sample': _gaussian_sample, 'bhattacharyya': _gaussian_bhattacharyya,
    },
    'gamma': {
        'label': 'Gama', 'domain': 'intensity', 'n_features': 1,
        'requires_enl': True, 'fit_methods': ('mle', 'mme'),
        'fit': _gamma_fit, 'logpdf': _gamma_logpdf, 'cdf': _gamma_cdf,
        'sample': _gamma_sample, 'bhattacharyya': _gamma_bhattacharyya,
    },
    'intensity_joint_distribution': {
        'label': 'Par de intensidades', 'domain': 'intensity', 'n_features': 2,
        'requires_enl': True, 'fit_methods': ('moments',),
        'fit': _pair_fit, 'logpdf': _pair_logpdf, 'cdf': None,
        'sample': _pair_sample, 'bhattacharyya': None,
    },
}


def get_distribution(name):
    """Retorna a entrada do catálogo, com mensagem clara se o nome não existir."""
    if name not in DISTRIBUTIONS:
        raise ValueError(f"Distribuição '{name}' não cadastrada. Disponíveis: {list(DISTRIBUTIONS)}")
    return DISTRIBUTIONS[name]


def check_data(name, x):
    """Confere se o nº de atributos de x é compatível com a distribuição."""
    dist = get_distribution(name)
    p = as_2d(x).shape[1]
    if dist['n_features'] is not None and p != dist['n_features']:
        raise ValueError(f"'{name}' ({dist['label']}) exige {dist['n_features']} atributo(s); recebidos {p}.")


def list_distributions():
    """Tabela-resumo do catálogo."""
    import pandas as pd
    rows = [{'name': k, 'label': d['label'], 'domain': d['domain'],
             'n_features': d['n_features'] or '≥ 1', 'requires_enl': d['requires_enl'],
             'fit_methods': ', '.join(d['fit_methods']), 'cdf (KS)': d['cdf'] is not None,
             'bhattacharyya': d['bhattacharyya'] is not None} for k, d in DISTRIBUTIONS.items()]
    return pd.DataFrame(rows)
