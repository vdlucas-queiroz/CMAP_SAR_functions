def cmap_classifier(tif_paths, weights_df, bands_config, output_dir,
                    date_labels=None, save_log_posterior=True, progress_step=10):
    """
    Classificador CMAP (Eq. 2.11): para cada pixel,
        ŝ = arg max_s  P(s) · Π_t P(o_t | ω_t^{k_t})
    calculado em espaço logarítmico (mesmo arg max, sem underflow numérico).

    Args:
        tif_paths (list): Rasters de verossimilhança, em ordem temporal
            (uma banda por classe).
        weights_df (pd.DataFrame): Saída de create_transition_matrix_table
            (colunas Stage_1..Stage_T e final_weight).
        bands_config (list of lists): Classe de cada banda, por data.
            Ex.: [['A','B'], ['A','B','C']] -> na data 2, banda 3 = 'C'.
        output_dir (str | Path): Pasta de saída.
        date_labels (list | None): Rótulos das datas para nomear os arquivos
            (ex.: ['2005','2006']). Se None, usa 1..T.
        save_log_posterior (bool): Salva o raster de log P(s, O) da trajetória vencedora.
        progress_step (int): Intervalo (%) das mensagens de progresso.

    Returns:
        dict: caminhos dos arquivos gerados, mapa de IDs e contagens por classe.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_dates = len(tif_paths)

    # ------------------------------------------------------------------
    # 0. Validações
    # ------------------------------------------------------------------
    if len(bands_config) != n_dates:
        raise ValueError(f"bands_config tem {len(bands_config)} datas, "
                         f"mas foram informados {n_dates} rasters.")
    if date_labels is None:
        date_labels = [str(d + 1) for d in range(n_dates)]
    if len(date_labels) != n_dates:
        raise ValueError("date_labels deve ter o mesmo tamanho de tif_paths.")

    stage_cols = [c for c in weights_df.columns if str(c).startswith('Stage_')]
    if len(stage_cols) != n_dates:
        raise ValueError(f"A tabela de pesos tem {len(stage_cols)} estágios, "
                         f"mas foram informados {n_dates} rasters.")
    if 'final_weight' not in weights_df.columns:
        raise ValueError("A tabela de pesos deve conter a coluna 'final_weight'.")

    bands_config = [[str(c).strip() for c in cfg] for cfg in bands_config]
    traj_names = weights_df[stage_cols].astype(str).apply(lambda s: s.str.strip()).to_numpy()
    weights = weights_df['final_weight'].to_numpy(dtype=np.float64)
    n_traj = len(weights_df)

    print("=== CMAP ===")

    # ------------------------------------------------------------------
    # 1. Dicionário global de IDs (saída)
    # ------------------------------------------------------------------
    unique_classes = sorted(set(traj_names.ravel()))
    if len(unique_classes) > 254:
        raise ValueError("Mais de 254 classes: não cabem em uint8 (0 = nodata).")
    global_id_map = {name: i + 1 for i, name in enumerate(unique_classes)}
    print(f"IDs de saída (0 = nodata): {global_id_map}")

    # ------------------------------------------------------------------
    # 2. Trajetória -> banda a ler em cada data / ID global de saída
    # ------------------------------------------------------------------
    traj_band_lookup = np.full((n_traj, n_dates), -1, dtype=np.int32)
    traj_global_id_lookup = np.zeros((n_traj, n_dates), dtype=np.uint8)
    missing = {}
    for t in range(n_traj):
        for d in range(n_dates):
            name = traj_names[t, d]
            if name in bands_config[d]:
                traj_band_lookup[t, d] = bands_config[d].index(name)
            else:
                missing.setdefault(date_labels[d], set()).add(name)
            traj_global_id_lookup[t, d] = global_id_map[name]

    for lbl, names in missing.items():
        print(f"Aviso: classes da tabela sem banda na data {lbl}: {sorted(names)} "
              f"-> trajetórias tratadas como P = 0.")

    valid_traj = (weights > 0) & (traj_band_lookup >= 0).all(axis=1)
    n_valid = int(valid_traj.sum())
    print(f"Trajetórias: {n_traj} no total | {n_valid} avaliadas | "
          f"{n_traj - n_valid} descartadas (P(s) = 0 ou classe sem banda)")
    if n_valid == 0:
        raise ValueError("Nenhuma trajetória válida para classificar.")

    with np.errstate(divide='ignore'):
        log_w = np.log(weights)

    # ------------------------------------------------------------------
    # 3. Leitura dos rasters (log-verossimilhança) e máscara de pixels válidos
    # ------------------------------------------------------------------
    print("Carregando rasters de verossimilhança...")
    log_lik = []
    valid_pixel = None
    for d, path in enumerate(tif_paths):
        with rasterio.open(path) as src:
            if d == 0:
                height, width = src.height, src.width
                crs, transform = src.crs, src.transform
                valid_pixel = np.ones((height, width), dtype=bool)
            elif (src.height, src.width) != (height, width):
                raise ValueError(f"'{path}' tem dimensões diferentes do primeiro raster.")
            if src.count != len(bands_config[d]):
                raise ValueError(f"'{path}' tem {src.count} bandas, mas bands_config "
                                 f"indica {len(bands_config[d])}.")

            data = src.read().astype(np.float64)
            bad = ~np.isfinite(data)
            if src.nodata is not None:
                bad |= (data == src.nodata)
            n_neg = int(((data < 0) & ~bad).sum())
            if n_neg > 0:
                print(f"Aviso: {n_neg} valores negativos em '{Path(path).name}' "
                      f"-> pixels mascarados.")
                bad |= (data < 0)
            valid_pixel &= ~bad.any(axis=0)

            with np.errstate(divide='ignore', invalid='ignore'):
                ll = np.log(data)
            ll[bad] = -np.inf
            log_lik.append(ll.astype(np.float32))

    print(f"Pixels válidos: {int(valid_pixel.sum())} de {height * width}")

    # ------------------------------------------------------------------
    # 4. arg max sobre as trajetórias
    # ------------------------------------------------------------------
    max_log_post = np.full((height, width), -np.inf, dtype=np.float64)
    best_traj = np.full((height, width), -1, dtype=np.int32)

    idx_valid = np.flatnonzero(valid_traj)
    next_report = progress_step
    for k, s in enumerate(idx_valid):
        acc = np.full((height, width), log_w[s], dtype=np.float64)
        for d in range(n_dates):
            acc += log_lik[d][traj_band_lookup[s, d]]

        mask = acc > max_log_post          # empate: mantém a primeira trajetória
        max_log_post[mask] = acc[mask]
        best_traj[mask] = s

        pct = 100 * (k + 1) / len(idx_valid)
        if pct >= next_report:
            print(f"  Progresso: {pct:5.1f}% ({k + 1}/{len(idx_valid)} trajetórias)")
            next_report += progress_step

    best_traj[~valid_pixel] = -1
    n_unassigned = int(((best_traj < 0) & valid_pixel).sum())
    if n_unassigned > 0:
        print(f"Aviso: {n_unassigned} pixels válidos com P = 0 para todas as "
              f"trajetórias -> nodata.")
    classified = best_traj >= 0

    # ------------------------------------------------------------------
    # 5. Saídas
    # ------------------------------------------------------------------
    print("Gravando saídas...")
    base_profile = {'driver': 'GTiff', 'height': height, 'width': width, 'count': 1,
                    'crs': crs, 'transform': transform, 'compress': 'deflate'}
    outputs = {'classified': [], 'global_id_map': global_id_map}
    safe_idx = np.where(classified, best_traj, 0)

    counts = {}
    for d in range(n_dates):
        final_map = np.where(classified, traj_global_id_lookup[safe_idx, d], 0).astype(np.uint8)
        out_name = output_dir / f"Classified_{date_labels[d]}.tif"
        with rasterio.open(out_name, 'w', dtype='uint8', nodata=0, **base_profile) as dst:
            dst.write(final_map, 1)
        outputs['classified'].append(str(out_name))
        vals, cnt = np.unique(final_map[classified], return_counts=True)
        inv = {v: k for k, v in global_id_map.items()}
        counts[date_labels[d]] = {inv[v]: int(c) for v, c in zip(vals, cnt)}

    out_name = output_dir / "Trajectory_IDs.tif"
    with rasterio.open(out_name, 'w', dtype='int32', nodata=-1, **base_profile) as dst:
        dst.write(best_traj, 1)
    outputs['trajectory_ids'] = str(out_name)

    if save_log_posterior:
        lp = np.where(classified, max_log_post, np.nan).astype(np.float32)
        out_name = output_dir / "Log_Posterior.tif"
        with rasterio.open(out_name, 'w', dtype='float32', nodata=np.nan, **base_profile) as dst:
            dst.write(lp, 1)
        outputs['log_posterior'] = str(out_name)

    # Legenda de classes
    legend_path = output_dir / 'legend_keys.csv'
    pd.DataFrame(list(global_id_map.items()), columns=['Class', 'ID']).to_csv(
        legend_path, index=False, sep=';')
    outputs['legend'] = str(legend_path)

    # Chave das trajetórias (Trajectory_ID = índice da linha em weights_df)
    ids, n_pix = np.unique(best_traj[classified], return_counts=True)
    traj_keys = weights_df[stage_cols + ['final_weight']].copy()
    traj_keys.insert(0, 'Trajectory_ID', np.arange(n_traj))
    traj_keys['n_pixels'] = 0
    traj_keys.loc[ids, 'n_pixels'] = n_pix
    traj_keys_path = output_dir / 'trajectory_keys.csv'
    traj_keys.to_csv(traj_keys_path, index=False, sep=';')
    outputs['trajectory_keys'] = str(traj_keys_path)

    # Contagem de pixels por classe e data
    class_counts = pd.DataFrame(counts).reindex(unique_classes).fillna(0).astype(int)
    class_counts.index.name = 'Class'
    counts_path = output_dir / 'class_counts.csv'
    class_counts.to_csv(counts_path, sep=';')
    outputs['class_counts'] = class_counts

    print("\nPixels por classe e data:")
    print(class_counts.to_string())
    print("\nTrajetórias mais frequentes:")
    print(traj_keys[traj_keys['n_pixels'] > 0]
          .sort_values('n_pixels', ascending=False).head(10).to_string(index=False))
    print(f"\nConcluído. Saídas em: {output_dir}")

    return outputs
