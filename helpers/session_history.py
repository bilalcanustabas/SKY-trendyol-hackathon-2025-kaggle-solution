import polars as pl


def candidate_counter(
    df: pl.DataFrame,
    session_col: str = "session_id",
    content_col: str = "content_id_hashed",
) -> pl.DataFrame:
    candidate_counter = df.group_by(session_col).agg(pl.count(content_col)).rename({content_col: "session_candidate_count"})
    df = df.join(
        candidate_counter,
        on=session_col,
        how="left"
    )
    return df


def session_based_ranking_for_contents(
    df: pl.DataFrame,
    session_col: str = "session_id",
    sitewide_table: str = "content_sitewide",
    tables: list[str] = ["content_top_terms","content_sitewide","content_search"],
    cols_search: list[str] = ["total_search_impression","total_search_click"],
    cols_sitewide: list[str] = ["total_click","total_cart","total_order","total_fav"],
    avg_ratio_template: str = "{table}_{col1}_to_{col2}_avg_ratio",
    non_agg_col_template: str = "{table}_{col}",
    weighted_col_template: str = "{table}_weighted_score",
    weights: dict[str, float] = {
        "total_order": 0.476,
        "total_click": 0.204,
        "total_cart": 0.254,
        "total_fav": 0.066,
        "total_search_impression": 0.1,
        "total_search_click": 0.9
    },
    table_weights: dict[str, float] = {
        "content_top_terms": 0.3,
        "content_sitewide": 0.6,
        "content_search": 0.1
    },
    price_columns: list[str] = ["original_price","selling_price","discounted_price"]
) -> pl.DataFrame:

    # 1- mevcut content interaction kolonlarinin bulnumasi
    df_cols = df.columns
    existing_cols = []

    for table in tables:
        if table == sitewide_table:
            cols = cols_sitewide
        else:
            cols = cols_search
        for col1 in cols:
            for col2 in cols:
                if col1 != col2:
                    template = {"table": table, "col1": col1, "col2": col2}
                    avg_ratio_col = avg_ratio_template.format(**template)
                    if avg_ratio_col in df_cols:
                        existing_cols.append(avg_ratio_col)
                    template["col"] = col1
                    non_agg_col1 = non_agg_col_template.format(**template)
                    if non_agg_col1 in df_cols:
                        existing_cols.append(non_agg_col1)
                    template["col"] = col2
                    non_agg_col2 = non_agg_col_template.format(**template)
                    if non_agg_col2 in df_cols:
                        existing_cols.append(non_agg_col2)

    # 2- mevcut content interaction kolonlarinin ranking'lerinin olusturulmasi
    for col in existing_cols:
        df = df.with_columns(
            (-pl.col(col)).rank(method="min").over(partition_by=session_col).alias(f"rank_{session_col}_{col}")
        )

    # 3- target weightleriyle content table'larinin kendi icinde weighted score ve ranking'lerinin olusturulmasi
    weighted_rank_cols = []
    for table in tables:
        if table == sitewide_table:
            cols = cols_sitewide
        else:
            cols = cols_search
        weighted_col = weighted_col_template.format(**{"table":table})
        df = df.with_columns(
            pl.sum_horizontal([pl.col(f"{table}_{col}") * weights[col] for col in cols]).alias(weighted_col)
        )
        weighted_rank_col = f"rank_{session_col}_{weighted_col}"
        df = df.with_columns(
            (-pl.col(weighted_col)).rank(method="min").over(partition_by=session_col).alias(weighted_rank_col)
        )
        weighted_rank_cols.append(weighted_rank_col)

    # 4- avg rank, median rank, total weighted score olusturulmasi
    rank_cols = weighted_rank_cols + [f"rank_{session_col}_{col}" for col in existing_cols]

    df = df.with_columns(
        (pl.mean_horizontal(rank_cols)).alias("avg_content_search_and_sitewide_rank"),
        (pl.concat_list(rank_cols).list.median()).alias("median_content_search_and_sitewide_rank"),
        (pl.sum_horizontal([pl.col(weighted_col_template.format(**{"table":table})) * table_weights[table] for table in tables])).alias("total_content_search_and_sitewide_weighted_score")
    )

    # 5- price ve content review ile alakali rankingler
    low_rank_cols = [f"{col}_log" for col in price_columns]
    high_rank_cols = ["discount_rate","selling_rate","content_rate_avg_bayesian","content_review_count_norm","content_review_wth_media_count_norm"]
    high_rank_cols += ["wilson_score_rate_to_review","wilson_score_review_to_media"]

    for col in low_rank_cols:
        df = df.with_columns(
            pl.col(col).rank(method="min").over(partition_by=session_col).alias(f"rank_{session_col}_{col}")
        )

    for col in high_rank_cols:
        df = df.with_columns(
            (-pl.col(col)).rank(method="min").over(partition_by=session_col).alias(f"rank_{session_col}_{col}")
        )

    return df