def create_transition_matrix_table(file_paths, output_path, sep=';', decimal='.',
                                   initial_probs=None, tol=1e-6):
    """
    Gera a tabela com todas as trajetórias possíveis (produto cartesiano das classes)
    e calcula a probabilidade a priori de cada trajetória, P(s) (Eq. 2.12).

    Args:
        file_paths (list): Caminhos das matrizes de transição, em ordem temporal
            (t1->t2, t2->t3, ...). Linhas = classe em t-1; colunas = classe em t.
            A primeira coluna do arquivo deve conter os nomes das classes de origem.
        output_path (str | Path): Caminho do CSV de saída.
        sep (str): Separador de colunas dos arquivos de entrada (padrão ';').
        decimal (str): Separador decimal dos arquivos de entrada (padrão '.').
        initial_probs (dict | None): Vetor de probabilidade inicial P(ω1), ex.
            {'A': 0.4, 'B': 0.6}. Se None, P(ω1) é considerado uniforme e omitido
            (não altera o arg max).
        tol (float): Tolerância para verificar se as linhas somam 1 (Eq. 2.13).

    Returns:
        pd.DataFrame: Colunas Stage_1..Stage_T, weight_0 (se initial_probs),
                      weight_1..weight_{T-1} e final_weight.
    """
    if len(file_paths) == 0:
        raise ValueError("Informe ao menos uma matriz de transição.")

    # 1. Leitura e padronização das matrizes
    matrices = []
    for path in file_paths:
        df = pd.read_csv(path, sep=sep, index_col=0, decimal=decimal)
        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()

        if df.index.duplicated().any() or df.columns.duplicated().any():
            raise ValueError(f"Classes duplicadas na matriz '{path}'.")
        try:
            df = df.astype(float)
        except ValueError as err:
            raise ValueError(f"Valores não numéricos na matriz '{path}' "
                             f"(verifique 'sep' e 'decimal'): {err}")
        if (df.values < 0).any():
            raise ValueError(f"A matriz '{path}' contém valores negativos.")

        row_sums = df.sum(axis=1)
        not_one = row_sums[(row_sums - 1.0).abs() > tol]
        if len(not_one) > 0:
            print(f"Aviso: em '{Path(path).name}', {len(not_one)} linha(s) não somam 1 "
                  f"(Eq. 2.13): {not_one.round(4).to_dict()}. "
                  f"Esperado apenas se for matriz de validade (abordagem simplificada).")
        matrices.append(df)

    # 2. Consistência entre estágios: colunas da matriz i == linhas da matriz i+1
    for i in range(len(matrices) - 1):
        cols_i = set(matrices[i].columns)
        rows_next = set(matrices[i + 1].index)
        if cols_i != rows_next:
            raise ValueError(
                f"Inconsistência no Stage_{i + 2}: destino da matriz {i + 1} "
                f"{sorted(cols_i)} ≠ origem da matriz {i + 2} {sorted(rows_next)}.")

    # 3. Classes por estágio e produto cartesiano
    classes_per_stage = [list(matrices[0].index)] + [list(m.columns) for m in matrices]
    stage_cols = [f'Stage_{i + 1}' for i in range(len(classes_per_stage))]
    df_final = pd.DataFrame(list(itertools.product(*classes_per_stage)), columns=stage_cols)

    # 4. Pesos (busca vetorizada por nome de linha/coluna)
    weight_cols = []
    if initial_probs is not None:
        init = {str(k).strip(): float(v) for k, v in initial_probs.items()}
        missing = set(classes_per_stage[0]) - set(init)
        if missing:
            raise ValueError(f"initial_probs não contém as classes: {sorted(missing)}")
        df_final['weight_0'] = df_final['Stage_1'].map(init).astype(float)
        weight_cols.append('weight_0')

    for i, m in enumerate(matrices):
        rows = m.index.get_indexer(df_final[f'Stage_{i + 1}'])
        cols = m.columns.get_indexer(df_final[f'Stage_{i + 2}'])
        col_name = f'weight_{i + 1}'
        df_final[col_name] = m.to_numpy()[rows, cols]
        weight_cols.append(col_name)

    # 5. Probabilidade a priori da trajetória P(s)
    df_final['final_weight'] = df_final[weight_cols].prod(axis=1)

    # 6. Salvar e resumir
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, sep=';', index=False)

    n_valid = int((df_final['final_weight'] > 0).sum())
    print(f"Tabela de trajetórias salva em: {output_path}")
    print(f"  Estágios: {len(stage_cols)} | Classes por estágio: "
          f"{[len(c) for c in classes_per_stage]}")
    print(f"  Trajetórias: {len(df_final)} no total | {n_valid} com P(s) > 0 | "
          f"{len(df_final) - n_valid} inválidas (P(s) = 0)")
    if initial_probs is None:
        print("  P(ω1) não informado: considerado uniforme (omitido do produto).")

    return df_final
