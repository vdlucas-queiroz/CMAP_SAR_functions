# Testes

Execute dentro desta pasta. Os arquivos `originais_*` e `r11.R`/`r12.R`/`r13.R` são cópias dos códigos originais (os `r*.R` sem as linhas `library(...)`).

| Teste | Comando | Depende do R | Esperado |
|---|---|---|---|
| Coerência do catálogo de distribuições | `python check_catalogo.py` | não | todos OK |
| Scripts 11, 12, 13 | `python gen.py` → `Rscript run_r.R` → `python compare_python_R.py` | caret, dplyr, reshape2, fitdistrplus | 27 de 27 |
| 14, SAMPLES_TO_POINTS, trajectory_assessment, cloud_masking | `python gen_amostragem_trajetorias.py` → `Rscript run_r_amostragem_trajetorias.R` → `python compare_python_R_amostragem_trajetorias.py` | raster, sp, sf | 13 de 13 |
| correlation_2.ipynb | `python test_autocorr.py` | não (executa as funções originais do notebook) | 80 de 80; lags finais idênticos |
| CMAP | `python test_cmap.py` | não (compara com `referencia_cmap_v1/`) | tudo `True`; 0% de trajetórias impossíveis no CMAP |
| CMAP com máscara por data × `CMAP_classifier` do R | `python test_cmap_vs_R.py` | raster, terra | `True` |
| Duplicados e extração em polígonos | `python test_amostras_vs_R.py` | dplyr, raster, sf | `True` |

Notas:
- `rgdal::writeOGR` foi retirado do CRAN. Em `run_r_amostragem_trajetorias.R`, só a gravação foi substituída por CSV; a seleção de pixels é a do código original.
- A divisão aleatória usa geradores diferentes no R e no Python, por isso compara-se o **nº** de amostras por classe; a amostragem por lags é determinística e é comparada **pixel a pixel**.
- Tolerâncias dos testes 11–13: ~1e-12 nos cálculos fechados; 1e-3 nos parâmetros da Gama por MV (otimizadores diferentes); 1e-6 nos p-valores KS assintóticos.
