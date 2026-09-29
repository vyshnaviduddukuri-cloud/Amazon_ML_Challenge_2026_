#!/usr/bin/env python3
"""
Step 5 — Threshold Tuning
Evaluates validation candidates across thresholds to maximize the F0.5 (and F1) score.
"""

from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein
import re

CANDIDATE_DIR = Path("dataset/candidates")
MODEL_DIR = Path("dataset/models")
TRAIN_DIR = Path("dataset/train")


def extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))


def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    s1_names = df["s1_name"].to_list()
    c_names = df["match_name"].to_list()
    s1_cores = df["s1_name_core"].to_list()
    c_cores = df["match_name_core"].to_list()
    s1_addrs = df["s1_address"].to_list()
    c_addrs = df["match_address"].to_list()
    s1_countries = df["s1_country"].to_list()
    c_countries = df["match_country"].to_list()
    block_types = df["block_types"].to_list()

    jw_name, jw_core, lev_name = [], [], []
    tok_jaccard_name, tok_jaccard_addr = [], []
    num_exact_addr, country_exact, block_count = [], [], []

    for s1_n, cn, s1_c, cc, s1_a, ca, s1_ctry, c_ctry, bt in zip(
        s1_names,
        c_names,
        s1_cores,
        c_cores,
        s1_addrs,
        c_addrs,
        s1_countries,
        c_countries,
        block_types,
    ):
        s1_n, cn = s1_n or "", cn or ""
        s1_c, cc = s1_c or "", cc or ""
        s1_a, ca = s1_a or "", ca or ""

        jw_name.append(JaroWinkler.similarity(s1_n, cn))
        jw_core.append(JaroWinkler.similarity(s1_c, cc))
        lev_name.append(Levenshtein.normalized_similarity(s1_n, cn))

        t1_name, t2_name = set(s1_n.split()), set(cn.split())
        tok_jaccard_name.append(
            len(t1_name & t2_name) / len(t1_name | t2_name)
            if (t1_name or t2_name)
            else 0.0
        )

        t1_addr, t2_addr = set(s1_a.split()), set(ca.split())
        tok_jaccard_addr.append(
            len(t1_addr & t2_addr) / len(t1_addr | t2_addr)
            if (t1_addr or t2_addr)
            else 0.0
        )

        n1_addr, n2_addr = extract_numbers(s1_a), extract_numbers(ca)

        num_exact_addr.append(
            1.0
            if (n1_addr and n2_addr and (n1_addr & n2_addr))
            else (0.5 if not (n1_addr and n2_addr) else 0.0)
        )

        country_exact.append(
            1.0
            if (s1_ctry == c_ctry and s1_ctry != "")
            else 0.0
        )

        block_count.append(
            float(len(bt.split(",")) if bt else 1.0)
        )

    return df.with_columns([
        pl.Series("feat_jw_name", jw_name, dtype=pl.Float32),
        pl.Series("feat_jw_core", jw_core, dtype=pl.Float32),
        pl.Series("feat_lev_name", lev_name, dtype=pl.Float32),
        pl.Series("feat_tok_jaccard_name", tok_jaccard_name, dtype=pl.Float32),
        pl.Series("feat_tok_jaccard_addr", tok_jaccard_addr, dtype=pl.Float32),
        pl.Series("feat_num_exact_addr", num_exact_addr, dtype=pl.Float32),
        pl.Series("feat_country_exact", country_exact, dtype=pl.Float32),
        pl.Series("feat_block_count", block_count, dtype=pl.Float32),
    ])


def main():
    print("[1/3] Loading trained LightGBM model...")

    booster = lgb.Booster(
        model_file=str(MODEL_DIR / "lgbm_matcher.txt")
    )

    print(
        "[2/3] Loading validation partition "
        "(candidate_part_003.parquet)..."
    )

    val_part = pl.read_parquet(
        CANDIDATE_DIR
        / "candidate_partitions"
        / "candidate_part_003.parquet"
    )

    # Load lookups
    s1 = pl.read_parquet(
        CANDIDATE_DIR / "slim_parquet" / "train_s1.parquet"
    ).select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("normalized_name").alias("s1_name"),
        pl.col("normalized_name_core").alias("s1_name_core"),
        pl.col("normalized_address").alias("s1_address"),
        pl.col("normalized_country").alias("s1_country"),
    ])

    s2 = pl.read_parquet(
        CANDIDATE_DIR / "slim_parquet" / "train_s2.parquet"
    ).select([
        pl.col("entity_id").alias("candidate_entity_id"),
        pl.col("normalized_name").alias("match_name"),
        pl.col("normalized_name_core").alias("match_name_core"),
        pl.col("normalized_address").alias("match_address"),
        pl.col("normalized_country").alias("match_country"),
    ])

    s3 = pl.read_parquet(
        CANDIDATE_DIR / "slim_parquet" / "train_s3.parquet"
    ).select([
        pl.col("entity_id").alias("candidate_entity_id"),
        pl.col("normalized_name").alias("match_name"),
        pl.col("normalized_name_core").alias("match_name_core"),
        pl.col("normalized_address").alias("match_address"),
        pl.col("normalized_country").alias("match_country"),
    ])

    other = pl.concat([s2, s3], how="vertical")

    gt = (
        pl.scan_csv(
            TRAIN_DIR / "train_ground_truth.tsv",
            separator="\t"
        )
        .filter(
            pl.col("matched_entity_ids").fill_null("") != ""
        )
        .with_columns(
            pl.col("matched_entity_ids")
            .str.split(",")
            .alias("_ids")
        )
        .explode("_ids")
        .with_columns(
            pl.col("_ids")
            .str.strip_chars()
            .alias("candidate_entity_id")
        )
        .select([
            "source1_entity_id",
            "candidate_entity_id"
        ])
        .unique()
        .with_columns(
            pl.lit(1).alias("is_match")
        )
        .collect()
    )

    df_val = (
        val_part
        .join(
            s1,
            on="source1_entity_id",
            how="left"
        )
        .join(
            other,
            on="candidate_entity_id",
            how="left"
        )
        .join(
            gt,
            on=[
                "source1_entity_id",
                "candidate_entity_id"
            ],
            how="left"
        )
        .with_columns(
            pl.col("is_match").fill_null(0)
        )
    )

    print(
        f"Validation candidates: {df_val.height:,} "
        f"(True matches: {df_val['is_match'].sum():,})"
    )

    df_val = compute_features(df_val)

    feature_cols = [
        c for c in df_val.columns
        if c.startswith("feat_")
    ]

    X_val = df_val.select(feature_cols).to_numpy()
    y_true = df_val["is_match"].to_numpy()

    print(
        "[3/3] Predicting probabilities and evaluating "
        "macro-F0.5..."
    )

    preds = booster.predict(X_val)

    # Keep the identifiers needed for entity-level evaluation.
    s1_ids = df_val["source1_entity_id"].to_list()
    candidate_ids = df_val["candidate_entity_id"].to_list()
    y_true_list = y_true.tolist()

    # Build:
    #   true_set[S1] = true candidate IDs
    #   candidate_set[S1] = all candidates in this validation partition
    true_set = {}
    candidate_set = {}

    for s1_id, cid, label in zip(
        s1_ids,
        candidate_ids,
        y_true_list
    ):
        candidate_set.setdefault(s1_id, set())

        if label == 1:
            true_set.setdefault(s1_id, set()).add(cid)

    # IMPORTANT:
    # Evaluate only S1 entities represented in this validation partition.
    eval_s1_ids = list(candidate_set.keys())

    def f05_for_entity(true_ids, predicted_ids):
        true_ids = set(true_ids)
        predicted_ids = set(predicted_ids)

        tp = len(true_ids & predicted_ids)
        fp = len(predicted_ids - true_ids)
        fn = len(true_ids - predicted_ids)

        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        denom = (0.25 * precision) + recall

        if denom == 0:
            return 0.0

        return 1.25 * precision * recall / denom

    print("\n" + "=" * 75)

    print(
        f"{'Threshold':<12}"
        f"{'Macro-F0.5':<15}"
        f"{'Macro-F1':<15}"
        f"{'Avg Pred/S1':<15}"
        f"{'S1 Count':<10}"
    )

    print("=" * 75)

    best_thresh = 0.50
    best_macro_f05 = -1.0

    thresholds = np.arange(
        0.20,
        0.951,
        0.05
    )

    for thresh in thresholds:
        predicted_by_s1 = {}

        for s1_id, cid, prob in zip(
            s1_ids,
            candidate_ids,
            preds
        ):
            if prob >= thresh:
                predicted_by_s1.setdefault(
                    s1_id,
                    set()
                ).add(cid)

        f05_scores = []
        f1_scores = []
        total_predictions = 0

        for s1_id in eval_s1_ids:
            true_ids = true_set.get(
                s1_id,
                set()
            )

            predicted_ids = predicted_by_s1.get(
                s1_id,
                set()
            )

            tp = len(
                true_ids & predicted_ids
            )

            fp = len(
                predicted_ids - true_ids
            )

            fn = len(
                true_ids - predicted_ids
            )

            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0
            )

            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0
            )

            f05_denom = (
                (0.25 * precision) + recall
            )

            f05 = (
                1.25 * precision * recall
                / f05_denom
                if f05_denom > 0
                else 0.0
            )

            f1_denom = precision + recall

            f1 = (
                2.0 * precision * recall
                / f1_denom
                if f1_denom > 0
                else 0.0
            )

            f05_scores.append(f05)
            f1_scores.append(f1)

            total_predictions += len(
                predicted_ids
            )

        macro_f05 = float(
            np.mean(f05_scores)
        )

        macro_f1 = float(
            np.mean(f1_scores)
        )

        avg_pred = (
            total_predictions
            / len(eval_s1_ids)
        )

        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_thresh = float(thresh)

        print(
            f"{thresh:<12.2f}"
            f"{macro_f05:<15.4f}"
            f"{macro_f1:<15.4f}"
            f"{avg_pred:<15.3f}"
            f"{len(eval_s1_ids):<10}"
        )

    print("=" * 75)

    print(
        f"\nBest threshold for macro-F0.5: "
        f"{best_thresh:.2f}"
    )

    print(
        f"Best validation macro-F0.5: "
        f"{best_macro_f05:.4f}"
    )

    print(
        "\nNOTE: This threshold is estimated from "
        "candidate_part_003 only."
    )


if __name__ == "__main__":
    main()
