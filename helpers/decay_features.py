import polars as pl
import numpy as np


def add_decay_features_multiple(
    train_df: pl.DataFrame,
    interactions_df: pl.DataFrame,
    interaction_cols: list[str] = None,
    decay_life: int = 3,
    decay_value: float = np.log(0.5),
    rolling_windows: list[int] = [3,6,12],
    user_col: str = "user_id_hashed",
    content_col: str = "content_id_hashed",
    time_col: str = "ts_hour",
    alias: str = "fashion"
) -> pl.DataFrame:
    """
    train_df ve interactions_df tablolarından yarı ömür decay ile geçmiş etkileşim skorlarını ekler.
    """
    if interaction_cols is None:
        raise ValueError("`interaction_cols` hesaplanması gereken kolonları barındırmalıdır.")

    # 1 Tüm session step numaralarının hesaplanması
    interaction_steps = interactions_df.group_by([user_col, time_col]).agg(pl.count())
    train_steps = train_df.group_by([user_col, time_col]).agg(pl.count())
    
    steps = (
        pl.concat([interaction_steps, train_steps], how="vertical")
        .unique(subset=[user_col, time_col], keep="last")
        .sort([user_col, time_col]).with_columns(
        pl.cum_count(time_col).over(user_col).alias("interaction_step"))
        .drop("count")
    )
    steps = steps

    # 2 Step numaralarının birleştirilmesi
    interactions_df = interactions_df.join(
        steps,
        on=[user_col, time_col],
        how="left"
    ).sort([user_col, time_col])

    train_df = train_df.join(
        steps,
        on=[user_col, time_col],
        how="left"
    ).sort([user_col, time_col])

    # 3 Join (user + content üzerinden)
    joined = train_df.unique(subset=[user_col,content_col,time_col,"interaction_step"], keep="first").join(
        interactions_df,
        on=[user_col, content_col],
        how="left"
    )

    # 4 Sadece geçmiş interaksiyonları al
    joined = joined.filter(pl.col("interaction_step") > pl.col("interaction_step_right"))

    # 5 Step farkı
    joined = joined.with_columns(
        (pl.col("interaction_step") - pl.col("interaction_step_right")).alias("step_diff")
    )

    # 6 Decay hesaplama
    decay_factor = decay_value / decay_life
    joined = joined.with_columns(
        (pl.col("step_diff") * decay_factor).exp().alias("decay")
    )

    # 7 Weighted değerler
    weighted_cols = []
    for col in interaction_cols:
        if col in joined.collect_schema().names():
            w_col = f"{col}_weighted"
            joined = joined.with_columns((pl.col(col) * pl.col("decay")).alias(w_col))
            weighted_cols.append(w_col)

    # 8 rolling cols
    joined = joined.sort([user_col, "interaction_step_right"])
    rolling_cols = []
    for col in interaction_cols:
        if col in joined.collect_schema().names():
            for window in rolling_windows:
                weighted_col = f"{col}_weighted"
                rolling_col_mean_decayed = f"{weighted_col}_{window}roll_step_mean_{alias}"
                rolling_col_std_decayed = f"{weighted_col}_{window}roll_step_std_{alias}"
                rolling_col_mean = f"{col}_{window}roll_step_mean_{alias}"
                rolling_col_sum = f"{col}_{window}roll_step_sum_{alias}"
                joined = joined.with_columns(
                    (pl.col(weighted_col)
                    .rolling_mean(window_size=window, min_periods=1)
                    .over([user_col,content_col])
                    .alias(rolling_col_mean_decayed)),
                    (pl.col(weighted_col)
                    .rolling_std(window_size=window, min_periods=1)
                    .over([user_col,content_col])
                    .alias(rolling_col_std_decayed)),
                    (pl.col(col)
                    .rolling_mean(window_size=window, min_periods=1)
                    .over([user_col,content_col])
                    .alias(rolling_col_mean)),
                    (pl.col(col)
                    .rolling_sum(window_size=window, min_periods=1)
                    .over([user_col,content_col])
                    .alias(rolling_col_sum))
                )
                rolling_cols.extend([rolling_col_mean_decayed, rolling_col_std_decayed, rolling_col_mean, rolling_col_sum])

    # 9 Session bazında topla
    agg_df = joined.group_by(["interaction_step", user_col, content_col]).agg(
        *[pl.sum(c).alias(c.replace("_weighted", f"_decay_score_{alias}")) for c in weighted_cols],
        *[pl.last(c) for c in rolling_cols],
        *[pl.last(c).alias(f"{c}_{alias}") for c in interaction_cols]
    )
    decay_cols = [c for c in agg_df.collect_schema().names() if c.endswith(f"_decay_score_{alias}")]

    # 10 Ana tabloya geri ekle
    final_df = train_df.join(
        agg_df,
        on=["interaction_step", user_col, content_col],
        how="left"
    ).drop(["interaction_step"])

    final_df = final_df.with_columns([pl.col(c).fill_null(0) for c in decay_cols + rolling_cols])

    return final_df


def add_decay_features_single_key(
    train_df: pl.DataFrame,
    interactions_df: pl.DataFrame,
    interaction_cols: list[str] = None,
    decay_life: int = 3,
    decay_value: float = np.log(0.5),
    rolling_windows: list[int] = [3,6,12],
    user_col: str = "user_id_hashed",
    time_col: str = "ts_hour",
    alias: str = None
) -> pl.DataFrame:
    """
    train_df ve interactions_df tablolarından yarı ömür decay ile geçmiş etkileşim skorlarını ekler.
    """
    if interaction_cols is None:
        raise ValueError("`interaction_cols` hesaplanması gereken kolonları barındırmalıdır.")
    if alias is None:
        raise ValueError("`alias` isimlendirilmesi verilmek zorundadır.")

    # 1 Tüm session step numaralarının hesaplanması
    interaction_steps = interactions_df.group_by([user_col, time_col]).agg(pl.count())
    train_steps = train_df.group_by([user_col, time_col]).agg(pl.count())

    steps = (
        pl.concat([interaction_steps, train_steps], how="vertical")
        .unique(subset=[user_col, time_col], keep="last")
        .sort([user_col, time_col]).with_columns(
        pl.cum_count(time_col).over(user_col).alias("interaction_step"))
        .drop("count")
    )
    steps = steps

    # 2 Step numaralarının birleştirilmesi
    interactions_df = interactions_df.join(
        steps,
        on=[user_col, time_col],
        how="left"
    ).sort([user_col, time_col])

    train_df = train_df.join(
        steps,
        on=[user_col, time_col],
        how="left"
    ).sort([user_col, time_col])

    # 3 Join (user + content üzerinden)
    joined = train_df.unique(subset=[user_col,time_col,"interaction_step"], keep="first").join(
        interactions_df,
        on=user_col,
        how="left"
    )

    # 4 Sadece geçmiş interaksiyonları al
    joined = joined.filter(pl.col("interaction_step") > pl.col("interaction_step_right"))

    # 5 Step farkı
    joined = joined.with_columns(
        (pl.col("interaction_step") - pl.col("interaction_step_right")).alias("step_diff")
    )

    # 6 Decay hesaplama
    decay_factor = decay_value / decay_life
    joined = joined.with_columns(
        (pl.col("step_diff") * decay_factor).exp().alias("decay")
    )

    # 7 Weighted değerler
    weighted_cols = []
    for col in interaction_cols:
        if col in joined.collect_schema().names():
            w_col = f"{col}_weighted"
            joined = joined.with_columns((pl.col(col) * pl.col("decay")).alias(w_col))
            weighted_cols.append(w_col)

    # 8 rolling cols
    joined = joined.sort([user_col, "interaction_step_right"])
    rolling_cols = []
    for col in interaction_cols:
        if col in joined.collect_schema().names():
            for window in rolling_windows:
                weighted_col = f"{col}_weighted"
                rolling_col_mean_decayed = f"{weighted_col}_{window}roll_step_mean_{alias}"
                rolling_col_std_decayed = f"{weighted_col}_{window}roll_step_std_{alias}"
                rolling_col_mean = f"{col}_{window}roll_step_mean_{alias}"
                rolling_col_sum = f"{col}_{window}roll_step_sum_{alias}"
                joined = joined.with_columns(
                    (pl.col(weighted_col)
                    .rolling_mean(window_size=window, min_periods=1)
                    .over(user_col)
                    .alias(rolling_col_mean_decayed)),
                    (pl.col(weighted_col)
                    .rolling_std(window_size=window, min_periods=1)
                    .over(user_col)
                    .alias(rolling_col_std_decayed)),
                    (pl.col(col)
                    .rolling_mean(window_size=window, min_periods=1)
                    .over(user_col)
                    .alias(rolling_col_mean)),
                    (pl.col(col)
                    .rolling_sum(window_size=window, min_periods=1)
                    .over(user_col)
                    .alias(rolling_col_sum))
                )
                rolling_cols.extend([rolling_col_mean_decayed, rolling_col_std_decayed, rolling_col_mean, rolling_col_sum])

    # 9 Session bazında topla
    agg_df = joined.group_by(["interaction_step", user_col]).agg(
        [pl.sum(c).alias(c.replace("_weighted", f"_decay_score_{alias}")) for c in weighted_cols]+[pl.last(c) for c in rolling_cols]
    )

    decay_cols = [c for c in agg_df.collect_schema().names() if c.endswith(f"_decay_score_{alias}")]

    # 10 Ana tabloya geri ekle
    final_df = train_df.join(
        agg_df,
        on=["interaction_step",user_col],
        how="left"
    ).drop(["interaction_step"])

    final_df = final_df.with_columns([pl.col(c).fill_null(0) for c in decay_cols + rolling_cols])

    return final_df