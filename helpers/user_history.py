import polars as pl


def add_user_history(
    df: pl.DataFrame,
    user_df: pl.DataFrame,
    user_col: str = "user_id_hashed",
    time_col: str = "ts_hour",
    interaction_cols: list[str] = ["total_click", "total_fav", "total_cart", "total_order"],
    ratio_groups: list[tuple[str, str]] = [
        ("total_click", "total_order"), ("total_click", "total_cart"), 
        ("total_click", "total_fav"), ("total_cart", "total_order")],
    alias: str = "user_sitewide",
    weights: dict[str, float] = {"total_click": 0.204, "total_fav": 0.066, "total_cart": 0.254, "total_order": 0.476},
    exact_match: bool = False
) -> pl.DataFrame:
    """
    weights = {
        "ordered": 0.476,
        "clicked": 0.204,
        "added_to_cart": 0.254,
        "added_to_fav": 0.066
    }
    weights = {
        "search": 0.1,
        "click": 0.9
    }
    """
    _window_size = 10000

    # 1 - join_asof ve rolling işlemleri için sorting
    df = df.sort([user_col, time_col])
    user_df = user_df.sort([user_col, time_col])

    # 2 - rolling kolonlarının oluşturulması
    user_df = user_df.with_columns(
        *[pl.col(col).cum_sum().over(user_col).fill_null(0).alias(f"{alias}_{col}_sum") for col in interaction_cols],
        *[pl.col(col).cum_max().over(user_col).fill_null(0).alias(f"{alias}_{col}_max") for col in interaction_cols],
        *[pl.col(col).rolling_std(window_size=_window_size, min_periods=1).over(user_col).alias(f"{alias}_{col}_std") for col in interaction_cols],
        *[pl.when(pl.col(col) > 0).then(1).otherwise(0).cum_sum().over(user_col).fill_null(0).alias(f"{alias}_{col}_active_session_count") for col in interaction_cols],
        pl.col(time_col).cum_count().over(user_col).alias(f"{alias}_session_count")
    )
    user_df = user_df.with_columns(
        *[pl.when(pl.col(f"{alias}_{col}_sum") > 0).then(pl.col(f"{alias}_{col}_sum") / pl.col(f"{alias}_session_count")).otherwise(0).alias(f"{alias}_{col}_avg") for col in interaction_cols],
    )

    # 3 - ratio kolonlarının oluşturulması
    user_df = user_df.with_columns(
        *[(pl.when(pl.col(f"{alias}_session_count") > 0)
            .then(pl.col(f"{alias}_{col}_active_session_count") / pl.col(f"{alias}_session_count"))
            .otherwise(0)).alias(f"{alias}_{col}_active_session_ratio") for col in interaction_cols]
    )

    for col1, col2 in ratio_groups:
        user_df = user_df.with_columns((
            pl.when(pl.col(f"{alias}_{col1}_avg") > 0)
            .then(pl.col(f"{alias}_{col2}_avg") / pl.col(f"{alias}_{col1}_avg"))
            .otherwise(0).alias(f"{alias}_{col1}_to_{col2}_avg_ratio")
        ))

    # 4 - weighted score kolonlarının oluşturulması
    user_df = user_df.with_columns(
        (pl.lit(0).alias(f"{alias}_weighted_sum_score")),
        (pl.lit(0).alias(f"{alias}_weighted_avg_score"))
    )
    for col, weight in weights.items():
        user_df = user_df.with_columns(
            (pl.col(f"{alias}_weighted_sum_score") + (pl.col(f"{alias}_{col}_sum") * weight)).alias(f"{alias}_weighted_sum_score"),
            (pl.col(f"{alias}_weighted_avg_score") + (pl.col(f"{alias}_{col}_avg") * weight)).alias(f"{alias}_weighted_avg_score")
        )

    # 5 - df ile user_df'in birleştirilmesi
    user_df = user_df.rename({col:f"{alias}_{col}" for col in interaction_cols})
    df = df.join_asof(user_df, on=time_col, by=user_col, strategy="backward", allow_exact_matches=exact_match)
    df = df.fill_null(0)

    return df


def add_user_term_history(
    df: pl.DataFrame,
    user_df: pl.DataFrame,
    user_col = "user_id_hashed",
    time_col = "ts_hour",
    term_col = "search_term_normalized",
    interaction_cols = ["total_search_impression","total_search_click"],
    alias = "user_term_search",
    ratio_groups: list[tuple[str, str]] = [("total_search_impression", "total_search_click")],
    weights: dict[str, float] = {"total_search_click": 0.9, "total_search_impression": 0.1},
    exact_match: bool = False
) -> pl.DataFrame:

    # 1 - join_asof ve rolling işlemleri için sorting
    _window_size = 10000

    df = df.sort([user_col, term_col, time_col])
    user_df = user_df.sort([user_col, term_col, time_col])

    # 2 - rolling kolonlarının oluşturulması
    user_df = user_df.with_columns(
        *[pl.col(col).cum_sum().over([user_col,term_col]).fill_null(0).alias(f"{alias}_{col}_sum") for col in interaction_cols],
        *[pl.col(col).cum_max().over([user_col,term_col]).fill_null(0).alias(f"{alias}_{col}_max") for col in interaction_cols],
        *[pl.col(col).rolling_std(window_size=_window_size, min_periods=1).over([user_col,term_col]).alias(f"{alias}_{col}_std") for col in interaction_cols],
        *[pl.when(pl.col(col) > 0).then(1).otherwise(0).cum_sum().over([user_col,term_col]).fill_null(0).alias(f"{alias}_{col}_active_session_count") for col in interaction_cols],
        pl.col(time_col).cum_count().over([user_col,term_col]).alias(f"{alias}_session_count")
    )
    user_df = user_df.with_columns(
        *[pl.when(pl.col(f"{alias}_{col}_sum") > 0).then(pl.col(f"{alias}_{col}_sum") / pl.col(f"{alias}_session_count")).otherwise(0).alias(f"{alias}_{col}_avg") for col in interaction_cols],
    )

    # 3 - ratio kolonlarının oluşturulması
    user_df = user_df.with_columns(
        *[(pl.when(pl.col(f"{alias}_session_count") > 0)
            .then(pl.col(f"{alias}_{col}_active_session_count") / pl.col(f"{alias}_session_count"))
            .otherwise(0)).alias(f"{alias}_{col}_active_session_ratio") for col in interaction_cols]
    )

    for col1, col2 in ratio_groups:
        user_df = user_df.with_columns((
            pl.when(pl.col(f"{alias}_{col1}_avg") > 0)
            .then(pl.col(f"{alias}_{col2}_avg") / pl.col(f"{alias}_{col1}_avg"))
            .otherwise(0).alias(f"{alias}_{col1}_to_{col2}_avg_ratio")
        ))

    # 4 - weighted score kolonlarının oluşturulması
    user_df = user_df.with_columns(
        (pl.lit(0).alias(f"{alias}_weighted_sum_score")),
        (pl.lit(0).alias(f"{alias}_weighted_avg_score"))
    )
    for col, weight in weights.items():
        user_df = user_df.with_columns(
            (pl.col(f"{alias}_weighted_sum_score") + (pl.col(f"{alias}_{col}_sum") * weight)).alias(f"{alias}_weighted_sum_score"),
            (pl.col(f"{alias}_weighted_avg_score") + (pl.col(f"{alias}_{col}_avg") * weight)).alias(f"{alias}_weighted_avg_score")
        )

    # 5 - df ile user_df'in birleştirilmesi
    user_df = user_df.rename({col:f"{alias}_{col}" for col in interaction_cols})
    df = df.join_asof(user_df, on=time_col, by=[user_col, term_col], strategy="backward", allow_exact_matches=exact_match)
    df = df.fill_null(0)

    return df


def add_user_term_to_all_ratios(
    df: pl.DataFrame,
    interaction_cols: list[str] = ["total_search_impression", "total_search_click"],
    calculation_types: list[str] = ["sum", "avg", "max"],
    alias_all: str = "user_search",
    alias_terms: str = "user_top_terms",
    new_alias: str = "term_to_user"
) -> pl.DataFrame:

    # 1 - Oran kolonlarının eklenmesi
    for col in interaction_cols:
        for calc_type in calculation_types:
            col_name_all = f"{alias_all}_{col}_{calc_type}"
            col_name_terms = f"{alias_terms}_{col}_{calc_type}"
            df = df.with_columns(
                pl.when(pl.col(col_name_all) > 0).then(pl.col(col_name_terms) / pl.col(col_name_all)).otherwise(0).alias(f"{new_alias}_{col}_{calc_type}_ratio")
            )

    return df


def add_user_metadata(
    df: pl.DataFrame,
    user_metadata: pl.DataFrame,
    user_col: str
) -> pl.DataFrame:

    user_metadata = user_metadata.with_columns(
        (2025 - (pl.when(pl.col("user_birth_year").is_not_null())
        .then(pl.when(pl.col("user_birth_year") < 1960).then(1960).otherwise(pl.col("user_birth_year")))
        .otherwise(pl.col("user_birth_year").fill_null(pl.col("user_birth_year").quantile(0.5)))
        )).alias("user_age")
    )

    user_metadata = user_metadata.with_columns(
        pl.when(pl.col("user_age") - (pl.col("user_tenure_in_days")/365) < 16)
        .then(16)
        .otherwise(pl.col("user_age") - (pl.col("user_tenure_in_days")/365))
        .alias("user_sign_up_age"),
        pl.when(pl.col("user_age") - (pl.col("user_tenure_in_days")/365) < 16)
        .then(16 + (pl.col("user_tenure_in_days")/365))
        .otherwise(pl.col("user_age"))
        .alias("user_age")
    )

    user_metadata = user_metadata.with_columns((pl.col("user_age") - (pl.col("user_tenure_in_days")/365)).alias("user_sign_up_age"))

    user_metadata = user_metadata.select(["user_id_hashed","user_gender","user_age","user_sign_up_age","user_tenure_in_days"])

    df = df.join(
        user_metadata, 
        on=user_col,
        how="left"
    )

    return df