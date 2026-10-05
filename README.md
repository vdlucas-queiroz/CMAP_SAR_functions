# Projeto CMAP — processamento em Python

## Como usar

1. Ajuste `config_projeto.json`: caminhos, **ordem das bandas** de cada imagem, classes e cenários.
2. Para cada imagem (altere `DATA`):
   - `01_lags_amostragem.ipynb`: autocorrelação, lags e amostragem descorrelacionada;
   - `02_modelagem_classificacao.ipynb`: estatística, aderência, ENL, validação cruzada, representantes, verossimilhanças e MaxVer.
3. Para cada cenário (altere `CENARIO`): `03_cmap_avaliacao.ipynb`, com CMAP, MaxVer, trajetórias impossíveis, discordância e acurácia por data.

Cada notebook começa por **uma única célula de parâmetros**. O código das células é procedural e chama o pacote `sar_classification`.

## Organização das saídas

```text
<raiz_resultados>/
├── <data>/
│   ├── 01_lags_amostragem.xlsx        lags por polígono, resumo, lags finais, nº de amostras, parâmetros
│   ├── amostras.gpkg                  camadas 'treino' e 'validacao'
│   ├── 02_modelagem.xlsx              estatísticas, aderência, ENL, separabilidade, validação, representantes, matrizes
│   ├── modelos.json                   modelos representantes + ENL + sementes
│   ├── maxver_modelos.tif             1 banda por modelo
│   └── verossimilhanca/<modelo>.tif   log-verossimilhança, 1 banda por classe (entrada do CMAP)
└── cmap/<cenário>/
    ├── 03_cmap.xlsx                   trajetórias, contagens, impossíveis, discordância, avaliação
    ├── CMAP.tif, MaxVer.tif           1 banda por data
    └── discordancia.tif               (opcional; também Trajectory_IDs.tif, Log_Posterior.tif)
```

São 5 arquivos por imagem, mais 1 por modelo, e 4 por cenário. As figuras vão para um PDF por etapa (`SALVAR_FIGURAS = True`).

## Pastas

| Pasta | Conteúdo |
|---|---|
| `sar_classification/` | pacote (v2.2) |
| `exemplo/` | projeto fictício (`gerar_dados_ficticios.py`, `config_exemplo.json`) e os notebooks executados com ele |
| `documentacao/` | notebooks 00–07: referência de cada módulo, com fórmulas e exemplos |
| `testes/` | regressão contra os códigos originais (`COMO_RODAR.md`) |

## Dependências

`pip install -r requirements.txt` (versões testadas, Python 3.12). Para a pasta `testes/`, também é preciso o R com os pacotes listados no fim do arquivo.

