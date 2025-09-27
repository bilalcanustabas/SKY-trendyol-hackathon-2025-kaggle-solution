import polars as pl

def add_time_history(
    df: pl.DataFrame,
    df_value: pl.DataFrame,
    key_col: str = "user_id_hashed",
    index_col: str = "ts_hour",
    periods: list[str] = ["24h","72h"],
    cols: list[str] = ["total_click", "total_order", "total_cart", "total_fav"],
    ratio_cols: list[tuple[str,str]] = [("total_click", "total_order"), ("total_cart", "total_order"), ("total_fav", "total_order"), ("total_click", "total_cart"), ("total_click", "total_fav")],
    aggs: list[str] = ["mean","std","min","max","sum"], 
    ratio_aggs: list[str] = ["mean","std","sum"],
    alias: str = "user_sitewide",
    exact_match: bool = True
):

    df = df.sort([index_col, key_col])
    df_value = df_value.sort([index_col, key_col])

    for period in periods:
        df_value = df_value.rolling(
            index_column=index_col,
            period=period,
            by=key_col,
            closed="left",
        ).agg([
            *[getattr(pl.col(col), agg)().alias(f"{alias}_rolling_{agg}_{col}_{period}") for col in cols for agg in aggs]
        ]).join(df_value, on=[key_col] + [index_col], how="right")

    for period in periods:
        for agg in ratio_aggs:
            for col1, col2 in ratio_cols:
                df_value = df_value.with_columns(
                    pl.when(pl.col(f"{alias}_rolling_{agg}_{col1}_{period}") > 0)
                    .then(pl.col(f"{alias}_rolling_{agg}_{col2}_{period}")/pl.col(f"{alias}_rolling_{agg}_{col1}_{period}"))
                    .otherwise(0)
                    .alias(f"{alias}_{col1}_to_{col2}_{agg}_{period}_ratio")
                )

    df_value = df_value.with_columns([pl.col(col).shift(1).over(key_col).alias(f"{alias}_{col}_lag1") for col in cols])
    df_value = df_value.drop(cols)

    df = df.join_asof(
        df_value, 
        on=index_col, 
        by=key_col, 
        strategy="backward", 
        allow_exact_matches=exact_match
    )

    return df