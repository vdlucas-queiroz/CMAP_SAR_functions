"""
Leitura/gravação de rasters e geração de rasters de verossimilhança (origem: script 11).
A ordem dos pixels é a de linha (a mesma do pacote raster do R).
"""
import numpy as np
import pandas as pd

from .models import class_likelihood


def raster_to_table(path, band_names=None):
    """Raster multibanda -> (DataFrame pixels x bandas, metadados). Nodata -> NaN."""
    import rasterio
    with rasterio.open(path) as src:
        data = src.read().astype(np.float64)
        meta = {'height': src.height, 'width': src.width, 'crs': src.crs,
                'transform': src.transform, 'nodata': src.nodata}
        descriptions = list(src.descriptions)
    n_bands = data.shape[0]
    if band_names is None:
        band_names = [d if d else f'band_{i + 1}' for i, d in enumerate(descriptions)]
    if len(band_names) != n_bands:
        raise ValueError(f"band_names tem {len(band_names)} nomes; o raster tem {n_bands} bandas.")
    table = pd.DataFrame(data.reshape(n_bands, -1).T, columns=band_names)
    if meta['nodata'] is not None:
        table = table.replace(meta['nodata'], np.nan)
    return table, meta


def table_to_raster(table, meta, output_path, dtype='float32', nodata=None, compress='deflate'):
    """Tabela (pixels x bandas) -> GeoTIFF; descrição de cada banda = nome da coluna."""
    import rasterio
    h, w = meta['height'], meta['width']
    if len(table) != h * w:
        raise ValueError(f"A tabela tem {len(table)} linhas; esperado {h * w} ({h} x {w}).")
    arr = table.to_numpy(dtype=np.float64).T.reshape(table.shape[1], h, w)
    if nodata is not None:
        arr = np.where(np.isnan(arr), nodata, arr)
    profile = {'driver': 'GTiff', 'height': h, 'width': w, 'count': table.shape[1], 'dtype': dtype,
               'crs': meta['crs'], 'transform': meta['transform'], 'nodata': nodata, 'compress': compress}
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(arr.astype(dtype))
        for i, name in enumerate(table.columns, start=1):
            dst.set_band_description(i, str(name))
    return str(output_path)


def likelihood_raster(image_path, model, output_path, band_names=None, log=False):
    """
    Imagem -> raster de verossimilhança (uma banda por classe, nome da classe na descrição).
    band_names: nomes das bandas da imagem; None usa os atributos do modelo, na ordem das bandas.
    Pixels com nodata/NaN em qualquer banda de entrada saem como NaN.

    log=False grava verossimilhança (formato do cmap_classifier atual); valores abaixo de
    ~1e-38 viram 0 em float32. log=True grava log-verossimilhança.
    """
    table, meta = raster_to_table(image_path, band_names or model['features'])
    valid = table.notna().all(axis=1).to_numpy()
    out = pd.DataFrame(np.nan, index=table.index, columns=model['classes'])
    if valid.any():
        out.loc[valid] = class_likelihood(table.loc[valid], model, log=log).to_numpy()
    n_zero = int((out.loc[valid] < np.finfo(np.float32).tiny).all(axis=1).sum()) if not log else 0
    if n_zero:
        print(f"Aviso: {n_zero} pixels com verossimilhança < 1e-38 em todas as classes "
              f"(viram 0 em float32). Considere log=True.")
    return table_to_raster(out, meta, output_path, dtype='float32', nodata=np.nan)


def classification_raster(image_path, model, output_path, class_codes=None, band_names=None):
    """
    Classificação MaxVer de uma imagem -> raster uint8 (0 = nodata) + legenda CSV ao lado.

    class_codes: {classe: código}. Padrão: 1..K na ordem de model['classes'] (convenção do
    maximum_likelihood_classifier do R). Para comparar com o CMAP pixel a pixel, use o MESMO
    class_codes nas duas funções (ou cmap.default_class_codes, que é a ordem alfabética).
    """
    from .models import classify
    table, meta = raster_to_table(image_path, band_names or model['features'])
    codes = class_codes or {c: i + 1 for i, c in enumerate(model['classes'])}
    miss = set(model['classes']) - set(codes)
    if miss:
        raise ValueError(f"class_codes não contém as classes: {sorted(miss)}")
    valid = table.notna().all(axis=1).to_numpy()
    out = np.zeros(len(table), dtype=np.uint8)
    if valid.any():
        out[valid] = classify(table.loc[valid], model).map(codes).to_numpy().astype(np.uint8)
    table_to_raster(pd.DataFrame({'class': out}), meta, output_path, dtype='uint8', nodata=0)
    legend = pd.DataFrame(sorted(codes.items(), key=lambda x: x[1]), columns=['Class', 'ID'])
    legend_path = str(output_path).rsplit('.', 1)[0] + '_legend.csv'
    legend.to_csv(legend_path, sep=';', index=False)
    return {'raster': str(output_path), 'legend': legend_path, 'class_codes': codes}


def read_legend(path):
    """Legenda CSV (Class;ID) -> {código: classe}."""
    df = pd.read_csv(path, sep=';')
    return {int(i): str(c) for c, i in zip(df['Class'], df['ID'])}


def verify_rasters(paths, bands_config=None):
    """
    Confere se os rasters podem ser combinados pixel a pixel (origem: CMAP_FUNCTIONS):
    dimensões, CRS, transformação afim (alinhamento) e, opcionalmente, nº de bandas
    contra bands_config. Returns: bool.
    """
    import rasterio
    from pathlib import Path
    infos = []
    for path in paths:
        with rasterio.open(path) as src:
            infos.append({'arquivo': Path(path).name, 'linhas': src.height, 'colunas': src.width,
                          'bandas': src.count, 'dtype': src.dtypes[0], 'nodata': src.nodata,
                          'crs': src.crs.to_string() if src.crs else None, 'transform': src.transform,
                          'descricoes': list(src.descriptions)})
    print(pd.DataFrame(infos).drop(columns=['transform']).to_string(index=False))
    ref, ok = infos[0], True
    for info in infos[1:]:
        if (info['linhas'], info['colunas']) != (ref['linhas'], ref['colunas']):
            print(f"Erro: '{info['arquivo']}' tem dimensões ({info['linhas']}, {info['colunas']}) "
                  f"≠ ({ref['linhas']}, {ref['colunas']})."); ok = False
        if info['crs'] != ref['crs']:
            print(f"Erro: '{info['arquivo']}' tem CRS diferente de '{ref['arquivo']}'."); ok = False
        if not info['transform'].almost_equals(ref['transform']):
            print(f"Erro: '{info['arquivo']}' não está alinhado (transform) com '{ref['arquivo']}'."); ok = False
    if bands_config is not None:
        if len(bands_config) != len(infos):
            print(f"Erro: bands_config tem {len(bands_config)} datas; foram informados {len(infos)} rasters."); ok = False
        else:
            for info, cfg in zip(infos, bands_config):
                if info['bandas'] != len(cfg):
                    print(f"Erro: '{info['arquivo']}' tem {info['bandas']} bandas; bands_config indica {len(cfg)}."); ok = False
    if ok:
        print(f"Sucesso: {len(infos)} rasters compatíveis ({ref['linhas']} linhas x {ref['colunas']} colunas).")
    return ok



def classification_stack(image_path, models, output_path, class_codes, band_names=None):
    """
    Classificação MaxVer da mesma imagem por vários modelos -> UM GeoTIFF uint8 multibanda
    (uma banda por modelo; descrição da banda = nome do modelo; 0 = nodata).

    models: {nome: modelo}; todos devem usar atributos presentes em band_names.
    class_codes: {classe: código}, o mesmo para todos os modelos.
    """
    from .models import classify
    table, meta = raster_to_table(image_path, band_names)
    out = pd.DataFrame(0, index=table.index, columns=list(models), dtype=np.uint8)
    for name, model in models.items():
        miss = set(model['classes']) - set(class_codes)
        if miss:
            raise ValueError(f"class_codes não contém as classes do modelo '{name}': {sorted(miss)}")
        cols = model['features']
        valid = table[cols].notna().all(axis=1).to_numpy()
        if valid.any():
            out.loc[valid, name] = classify(table.loc[valid, cols], model).map(class_codes).to_numpy().astype(np.uint8)
    return table_to_raster(out, meta, output_path, dtype='uint8', nodata=0)
