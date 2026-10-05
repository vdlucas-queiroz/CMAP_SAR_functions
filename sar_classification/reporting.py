"""
Gravação enxuta de resultados: uma planilha por etapa (uma aba por tabela) e, opcionalmente,
um PDF com todas as figuras da etapa — em vez de dezenas de CSVs e imagens soltas.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def save_workbook(path, sheets, index=False):
    """
    Grava {nome_da_aba: DataFrame} em um único .xlsx (substitui o arquivo, se existir).
    Nomes de aba são truncados em 31 caracteres (limite do Excel).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as xw:
        for name, df in sheets.items():
            pd.DataFrame(df).to_excel(xw, sheet_name=str(name)[:31], index=index)
    print(f"Planilha gravada: {path} ({len(sheets)} abas: {', '.join(sheets)})")
    return str(path)


def read_workbook(path, sheet=None):
    """Lê uma aba (ou todas, como dict) de uma planilha gravada por save_workbook."""
    return pd.read_excel(path, sheet_name=sheet)


def open_pdf(path):
    """
    Abre um PDF para receber as figuras da etapa (passe-o no parâmetro pdf= das funções de gráfico).
    Feche com .close() ao final. Returns: PdfPages.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return PdfPages(path)


def save_json(obj, path):
    """Grava dicionários com arrays NumPy em JSON legível."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        if isinstance(v, dict):
            return {str(k): conv(w) for k, w in v.items()}
        if isinstance(v, (list, tuple)):
            return [conv(w) for w in v]
        return v
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(conv(obj), ensure_ascii=False, indent=2), encoding='utf-8')
    return str(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
