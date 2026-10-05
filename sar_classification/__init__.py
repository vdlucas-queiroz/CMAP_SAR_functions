"""
sar_classification — modelagem estatística, classificação MaxVer e validação de dados
SAR/ópticos; geração de rasters de verossimilhança para o CMAP.

Módulos (dependências apontam sempre para cima nesta lista):
    distributions — catálogo de distribuições (interface única)
    samples       — conjunto amostral
    models        — modelo de classes, verossimilhança, MaxVer          (usa distributions, samples)
    raster_io     — rasters e rasters de verossimilhança                (usa models)
    separability  — Bhattacharyya, JM, seleção de atributos, hierarquia (usa models)
    validation    — matriz de confusão, métricas, validação cruzada     (usa models)
    fitting       — ajuste, KS, Mardia, ENL                             (usa distributions, samples)
    plots         — gráficos
    diagnostics   — verificação automática de distribuições do catálogo
    sampling      — desenho amostral espacial: raster de amostras -> pontos -> df_samples (14, SAMPLES_TO_POINTS)
    autocorrelation — autocorrelação espacial e lags de descorrelação (correlation_2)
    masking       — mascaramento de imagens (cloud_masking)
    transitions   — matrizes de transição e P(s)
    cmap          — classificador CMAP                                   (usa transitions)
    trajectory_assessment — trajetórias impossíveis e MaxVer × CMAP      (usa transitions)
    reporting     — planilhas por etapa, PDF de figuras, JSON
"""
from . import (distributions, samples, models, raster_io, separability, validation, fitting, plots,
               diagnostics, sampling, autocorrelation, masking, transitions, cmap, trajectory_assessment,
               reporting)

__version__ = '2.2.0'
