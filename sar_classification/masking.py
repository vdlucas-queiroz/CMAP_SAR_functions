"""Mascaramento de imagens (origem: cloud_masking.R)."""
import numpy as np


def mask_array(image, mask, value, mask_value=1):
    """
    Substitui, em todas as bandas, os pixels onde mask == mask_value por `value`.
    image: (bandas, linhas, colunas) ou (linhas, colunas); mask: (linhas, colunas).
    Pixels NaN na máscara não são alterados (no original, NA na máscara gerava erro).
    """
    img = np.array(image, dtype=np.float64 if np.isnan(value) else np.asarray(image).dtype, copy=True)
    m = np.asarray(mask)
    if img.shape[-2:] != m.shape:
        raise ValueError(f"Imagem {img.shape[-2:]} e máscara {m.shape} devem ter as mesmas linhas e colunas.")
    hit = (m == mask_value)
    img[..., hit] = value
    return img


def apply_mask(image_path, mask_path, output_path, value, mask_value=1, set_as_nodata=True):
    """
    Versão em arquivo de masking_replace. Além do nº de linhas e colunas (verificação do original),
    confere CRS e transformação. Com set_as_nodata=True, `value` é declarado como nodata da saída.
    """
    import rasterio
    with rasterio.open(image_path) as src, rasterio.open(mask_path) as msk:
        if (src.height, src.width) != (msk.height, msk.width):
            raise ValueError("A imagem e a máscara devem ter a mesma dimensão.")
        if src.crs != msk.crs or not src.transform.almost_equals(msk.transform):
            raise ValueError("A imagem e a máscara não estão na mesma grade (CRS/transform).")
        out = mask_array(src.read(), msk.read(1), value, mask_value)
        prof = src.profile.copy()
        prof.update(dtype=out.dtype, compress='deflate')
        for k in ('blockxsize', 'blockysize', 'tiled'):
            prof.pop(k, None)
        if set_as_nodata:
            prof['nodata'] = value
        n_masked = int((msk.read(1) == mask_value).sum())
    with rasterio.open(output_path, 'w', **prof) as dst:
        dst.write(out)
    print(f"{n_masked} pixels mascarados em {out.shape[0]} banda(s) -> {output_path}")
    return str(output_path)
