import polars as pl


def add_content_price_history(
    df: pl.DataFrame,
    content_price: pl.DataFrame,
    content_metadata: pl.DataFrame,
    content_col: str = "content_id_hashed",
    left_time_col: str = "date",
    right_time_col: str = "update_date",
    categories: list[str] = ["level1_category_name", "level2_category_name", "leaf_category_name"],
    bayesian_m: int = 30,
    psuedo_alpha: int = 1,
    psuedo_beta: int = 1,
    wilson_z: float = 1.96,
    exact_match: bool = False
) -> pl.DataFrame:

    _min_value = content_price.filter(pl.col("content_review_count")>0).select(pl.col("content_review_count").min()).collect().item()
    C = content_price.select(pl.col("content_rate_avg").mean()).collect().item()
    _renormalize_columns = ["original_price","selling_price","discounted_price","content_review_count","content_review_wth_media_count","content_rate_count"]
    _price_columns = ["original_price","selling_price","discounted_price"]

    df = df.sort([content_col,left_time_col])
    content_price = content_price.sort([content_col,right_time_col])

    # 1 - null degerlerin doldurulmasi
    content_metadata = content_metadata.with_columns(pl.col("cv_tags").fill_null(""))
    content_metadata = content_metadata.fill_null(0)

    # 2 - category_sizes olusturulmasi
    for group in categories:
        category_sizes = content_metadata.group_by(group).agg(pl.count()).rename({"count": f"{group}_size"})
        content_metadata = content_metadata.join(category_sizes, on=group, how="left")

    # 3 - category_metadata'nin content_price'a eklenmesi
    content_price = content_price.join(content_metadata, on=content_col, how="left")

    # 4 - null categorylerin unknown ile doldurulmasi
    for cat_col in categories:
        content_price = content_price.with_columns(pl.col(cat_col).fill_null("unknown").alias(cat_col))

    # 5 - price kolonlarinin olusturulmasi
    content_price = content_price.with_columns(
        pl.when(pl.col("original_price") > 0).then(1 - (pl.col("discounted_price") / pl.col("original_price"))).otherwise(0).alias("discount_rate"),
        pl.when(pl.col("original_price") > 0).then(1 - (pl.col("selling_price") / pl.col("original_price"))).otherwise(0).alias("selling_rate"),
        ((pl.col("selling_price") - pl.col("discounted_price")) / pl.col("selling_price")).alias("selling_discount_diff_ratio"),
    )

    # 6 - price ve count'larin normal sayilara geri donusturulmesi
    content_price = content_price.with_columns(*[(pl.col(col)/_min_value).alias(f"{col}_norm") for col in _renormalize_columns])

    # 7 - log price'larin olusturulmasi
    content_price = content_price.with_columns(*[((pl.col(f"{col}_norm") + 1).log()).alias(f"{col}_log") for col in _price_columns])

    # 8 - bayesian rate avg olusturulmasi
    content_price = content_price.with_columns(
        (
            (pl.col("content_rate_count") * pl.col("content_rate_avg") + bayesian_m * C) /
            (pl.col("content_rate_count") + bayesian_m)
        ).alias("content_rate_avg_bayesian")
    )

    # 9 - smoothed ratio for counts
    content_price = content_price.with_columns(
        (
            (pl.col("content_review_count") + psuedo_alpha) / (pl.col("content_rate_count") + psuedo_alpha + psuedo_beta)
        ).alias("content_rate_to_review_smoothed_ratio"),
        (
            (pl.col("content_review_wth_media_count") + psuedo_alpha) / (pl.col("content_review_count") + psuedo_alpha + psuedo_beta)
        ).alias("content_review_to_media_smoothed_ratio")
    )

    # 10 - wilson score for counts (wilson_score_rate_to_review, wilson_score_review_to_media)
    content_price = content_price.with_columns(
        pl.when(pl.col("content_rate_count") > 0).then(pl.col("content_review_count") / pl.col("content_rate_count")).otherwise(0).alias("wilson_p_rate_to_review"),
        pl.when(pl.col("content_review_count") > 0).then(pl.col("content_review_wth_media_count") / pl.col("content_review_count")).otherwise(0).alias("wilson_p_review_to_media")
    )

    wilson_list = [("wilson_p_rate_to_review", "content_rate_count", "wilson_score_rate_to_review"), ("wilson_p_review_to_media", "content_review_count", "wilson_score_review_to_media")]
    for p_col, n_col, alias in wilson_list:
        content_price = content_price.with_columns(
            pl.when(pl.col(n_col) > 0)
            .then(
                (
                    pl.col(p_col) + (wilson_z ** 2 / (2 * pl.col(n_col)))
                    - wilson_z * (
                        (
                            (pl.col(p_col) * (1 - pl.col(p_col)) / pl.col(n_col))
                            + (wilson_z ** 2 / (4 * pl.col(n_col) ** 2))
                        ) ** 0.5
                    )
                ) / (1 + (wilson_z ** 2 / pl.col(n_col)))
            )
            .otherwise(0)
            .alias(alias)
        )

    content_price = content_price.drop(["wilson_p_rate_to_review","wilson_p_review_to_media"])

    # 11 - category mean/std prices
    for cat_col in categories:

        agg_df = content_price.group_by(cat_col).agg(
            *[pl.col(f"{col}_log").mean().alias(f"{cat_col}_{col}_log_mean") for col in _price_columns],
            *[pl.col(f"{col}_log").std().alias(f"{cat_col}_{col}_log_std") for col in _price_columns]
        ).fill_null(0)

        content_price = content_price.join(agg_df, on=cat_col, how="left")

    for col in _price_columns:

        global_mean = content_price.select(pl.col(f"{col}_log").mean()).collect().item()
        global_std = content_price.select(pl.col(f"{col}_log").std()).collect().item()

        for cat_col in categories:

            n_col = f"{cat_col}_size"

            content_price = content_price.with_columns(
                (pl.col(f"{cat_col}_{col}_log_mean") * (pl.col(n_col)/(pl.col(n_col)+bayesian_m)) + global_mean * (pl.col(n_col)/(pl.col(n_col)+bayesian_m))).alias(f"smoothed_{cat_col}_{col}_mean"),
                (pl.col(f"{cat_col}_{col}_log_std") * (pl.col(n_col)/(pl.col(n_col)+bayesian_m)) + global_std * (pl.col(n_col)/(pl.col(n_col)+bayesian_m))).alias(f"smoothed_{cat_col}_{col}_std")
            )

    # 12 - rank features
    low_rank_cols = [f"{col}_log" for col in _price_columns]
    high_rank_cols = ["discount_rate","selling_rate","content_rate_avg_bayesian","content_review_count_norm","content_review_wth_media_count_norm"]
    high_rank_cols += ["wilson_score_rate_to_review","wilson_score_review_to_media"]

    for col in low_rank_cols:
        for cat_col in categories:
            content_price = content_price.with_columns(
                pl.col(col).rank(method="min").over(partition_by=cat_col).alias(f"rank_{cat_col}_{col}")
            )

    for col in high_rank_cols:
        for cat_col in categories:
            content_price = content_price.with_columns(
                (-pl.col(col)).rank(method="min").over(partition_by=cat_col).alias(f"rank_{cat_col}_{col}")
            )

    # 13 - df'e ekleme
    df = df.join_asof(content_price, left_on=left_time_col, right_on=right_time_col, by=content_col, strategy="backward", allow_exact_matches=exact_match)
    
    # 14 - date col'ların eklenmesi
    df = df.with_columns(
        (pl.col("ts_hour").cast(pl.Datetime("ms")) - pl.col("content_creation_date").cast(pl.Datetime("ms"))).dt.total_hours().alias("content_tenure_hours"),
        (pl.col("ts_hour").cast(pl.Datetime("ms")) - pl.col(right_time_col).cast(pl.Datetime("ms"))).dt.total_hours().alias("update_tenure_hours")
    )

    df = df.with_columns(
        (pl.col("update_tenure_hours") / pl.col("content_tenure_hours")).alias("update_to_content_tenure_ratio"),
    )

    df = df.with_columns(
        pl.when(pl.col("content_tenure_hours") < 0)
        .then(pl.col("update_tenure_hours"))
        .otherwise(pl.col("content_tenure_hours")).alias("content_tenure_hours")
    )

    df = df.with_columns(
        pl.col("update_tenure_hours").fill_null(0).alias("update_tenure_hours"),
        pl.col("content_tenure_hours").fill_null(0).alias("content_tenure_hours"),
        pl.col("update_to_content_tenure_ratio").fill_null(0).alias("update_to_content_tenure_ratio")
    )

    return df