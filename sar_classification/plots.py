"""Gráficos (origem: scripts 11 e 12). Funções sem cálculo além do necessário à figura."""
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy

from .samples import CLASS_COL, feature_columns


def _finish(pdf=None):
    """Mostra a figura e, se pdf (PdfPages de reporting.open_pdf) for informado, grava nela."""
    import matplotlib.pyplot as plt
    if pdf is not None:
        pdf.savefig(plt.gcf(), bbox_inches='tight')
    plt.show()


def boxplots_by_class(df_samples, classes, title=None, pdf=None):
    """Boxplot de cada atributo por classe (parte gráfica de data_analysis)."""
    import matplotlib.pyplot as plt
    feats = feature_columns(df_samples)
    fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 4), squeeze=False)
    for ax, f in zip(axes[0], feats):
        ax.boxplot([df_samples.loc[df_samples[CLASS_COL] == c, f].dropna() for c in classes])
        ax.set_xticks(range(1, len(classes) + 1), classes)
        ax.set_title(f, fontweight='bold')
        ax.set_xlabel('Classe')
        ax.grid(axis='y', color='#d3d3d3')
    if title:
        fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    _finish(pdf)


def plot_dendrogram(linkage_matrix, labels, title=None, ylabel='Distância de Bhattacharyya', pdf=None):
    """Dendrograma a partir da matriz de ligação (separability.hierarchical_clustering)."""
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(8, 5))
    hierarchy.dendrogram(linkage_matrix, labels=list(labels), ax=ax, color_threshold=0,
                         above_threshold_color='k')
    ax.set_title(title or '', fontweight='bold')
    ax.set_xlabel('Classes', fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    _finish(pdf)


def bubble_confusion_plot(confusion_matrix, colors=None, title=None, rename=None, pdf=None):
    """Bolhas da matriz de confusão: frequência relativa por coluna (Reference), vírgula decimal."""
    import matplotlib.pyplot as plt
    cm = confusion_matrix.rename(index=rename, columns=rename) if rename else confusion_matrix.copy()
    rel = cm / cm.sum(axis=0)
    ref, pred = np.meshgrid(np.arange(cm.shape[1]), np.arange(cm.shape[0]))
    colors = colors or {c: 'gray' for c in cm.columns}
    _, ax = plt.subplots(figsize=(1.3 * cm.shape[1] + 2, 1.3 * cm.shape[0] + 1))
    ax.scatter(ref.ravel(), pred.ravel(), s=rel.to_numpy().ravel() * 2500,
               c=[colors.get(cm.columns[j], 'gray') for j in ref.ravel()], alpha=0.6)
    for (i, j), v in np.ndenumerate(rel.to_numpy()):
        ax.text(j, i, f"{100 * v:.1f}%".replace('.', ','), ha='center', va='center', fontweight='bold')
    ax.set_xticks(range(cm.shape[1]), cm.columns, fontweight='bold')
    ax.set_yticks(range(cm.shape[0]), cm.index, fontweight='bold')
    ax.set_xlabel('Reference', fontweight='bold')
    ax.set_ylabel('Prediction', fontweight='bold')
    ax.set_xlim(-0.6, cm.shape[1] - 0.4)
    ax.set_ylim(-0.6, cm.shape[0] - 0.4)
    ax.grid(color='0.85')
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontweight='bold')
    _finish(pdf)


def plot_single_raster(path, band=1, title='Imagem raster', cmap='viridis', pdf=None):
    """Uma banda de um raster contínuo (ex.: verossimilhança, log-posterior); nodata mascarado."""
    import matplotlib.pyplot as plt
    import rasterio
    with rasterio.open(path) as src:
        data = np.ma.masked_invalid(src.read(band, masked=True).astype(float))
    plt.figure(figsize=(8, 6))
    img = plt.imshow(data, cmap=cmap)
    plt.colorbar(img, label='Valor do pixel')
    plt.title(title)
    plt.axis('off')
    _finish(pdf)


def plot_classification_results(paths, titles, legend_path, pdf=None):
    """
    Mapas classificados lado a lado, com cor fixa por classe (mesma cor = mesma classe em
    todas as datas) e legenda. legend_path: CSV 'Class;ID' (cmap_classifier ou classification_raster).
    """
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    legend = pd.read_csv(legend_path, sep=';').sort_values('ID')
    n = int(legend['ID'].max())
    base = plt.get_cmap('tab20' if n > 10 else 'tab10')
    cm = ListedColormap([base(i % base.N) for i in range(n)])
    cm.set_bad(alpha=0)
    norm = BoundaryNorm(np.arange(0.5, n + 1.5), n)
    fig, axes = plt.subplots(1, len(paths), figsize=(5 * len(paths), 5), squeeze=False)
    for ax, path, title in zip(axes[0], paths, titles):
        with rasterio.open(path) as src:
            ax.imshow(src.read(1, masked=True), cmap=cm, norm=norm, interpolation='nearest')
        ax.set_title(title)
        ax.axis('off')
    handles = [Patch(color=cm(norm(i)), label=f"{i} - {c}") for c, i in zip(legend['Class'], legend['ID'])]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 8), frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    _finish(pdf)



def plot_model_comparison(summary, metric='OA', label='Modelo', group=None, title=None, pdf=None):
    """
    Média e intervalo (ci_lw–ci_up) de OA ou F1 por modelo (tabela 'global' de validation.summarize_models).
    group: coluna opcional para comparar séries (ex.: 'Classificador' = MaxVer × CMAP).
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * summary[label].nunique() + 2), 4.5))
    groups = [(None, summary)] if group is None else list(summary.groupby(group, sort=False))
    xs = {m: i for i, m in enumerate(dict.fromkeys(summary[label]))}
    for k, (g, df) in enumerate(groups):
        off = 0 if group is None else (k - (len(groups) - 1) / 2) * 0.18
        x = np.array([xs[m] for m in df[label]]) + off
        y = df[f'{metric}_media'].to_numpy()
        err = np.vstack([y - df[f'{metric}_ci_lw'], df[f'{metric}_ci_up'] - y])
        ax.errorbar(x, y, yerr=err, fmt='o', capsize=4, label=g)
    ax.set_xticks(range(len(xs)), list(xs), rotation=45, ha='right')
    ax.set_ylabel({'OA': 'Acurácia Global', 'F1': 'F1-score macro'}.get(metric, metric))
    ax.grid(axis='y', color='0.85')
    if group is not None:
        ax.legend(title=group)
    ax.set_title(title or '')
    plt.tight_layout()
    _finish(pdf)


def plot_class_metrics(per_class, metric='PA', label='Modelo', title=None, pdf=None):
    """
    Média e intervalo de PA, UA ou F1 por classe e modelo (tabela 'classes' de validation.summarize_models).
    """
    import matplotlib.pyplot as plt
    classes = list(dict.fromkeys(per_class['Classe']))
    models = list(dict.fromkeys(per_class[label]))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(classes) + 2), 4.5))
    for k, m in enumerate(models):
        df = per_class[per_class[label] == m].set_index('Classe').reindex(classes)
        off = (k - (len(models) - 1) / 2) * min(0.8 / len(models), 0.15)
        y = df[f'Mean_{metric}'].to_numpy()
        err = np.vstack([y - df[f'ci_lw_{metric}'], df[f'ci_up_{metric}'] - y])
        ax.errorbar(np.arange(len(classes)) + off, y, yerr=err, fmt='o', capsize=3, label=m)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_ylabel({'PA': 'Acurácia do Produtor', 'UA': 'Acurácia do Usuário', 'F1': 'F1-score'}[metric])
    ax.grid(axis='y', color='0.85')
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title or '')
    plt.tight_layout()
    _finish(pdf)
