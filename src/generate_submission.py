#!/usr/bin/env python3
"""
Step 6 & 7 — Generate matching_results.tsv
Streams all 64 candidate partitions, runs feature extraction + LightGBM inference,
and writes out the final matching_results.tsv with format:
source1_entity_id \t matched_entity_ids
"""

from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein
import re
import gc

CANDIDATE_DIR = Path("dataset/candidates")
MODEL_DIR = Path("dataset/models")
OUTPUT_FILE = Path("matching_results.tsv")
TEMP_PARTS_DIR = Path("dataset/matching_parts")
TEMP_PARTS_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.95

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
        s1_names, c_names, s1_cores, c_cores, s1_addrs, c_addrs, s1_countries, c_countries, block_types
    ):
        s1_n, cn = s1_n or "", cn or ""
        s1_c, cc = s1_c or "", cc or ""
        s1_a, ca = s1_a or "", ca or ""

        jw_name.append(JaroWinkler.similarity(s1_n, cn))
        jw_core.append(JaroWinkler.similarity(s1_c, cc))
        lev_name.append(Levenshtein.normalized_similarity(s1_n, cn))

        t1_name, t2_name = set(s1_n.split()), set(cn.split())
        tok_jaccard_name.append(len(t1_name & t2_name) / len(t1_name | t2_name) if (t1_name or t2_name) else 0.0)

        t1_addr, t2_addr = set(s1_a.split()), set(ca.split())
        tok_jaccard_addr.append(len(t1_addr & t2_addr) / len(t1_addr | t2_addr) if (t1_addr or t2_addr) else 0.0)

        n1_addr, n2_addr = extract_numbers(s1_a), extract_numbers(ca)
        num_exact_addr.append(1.0 if (n1_addr and n2_addr and (n1_addr & n2_addr)) else (0.5 if not (n1_addr and n2_addr) else 0.0))
        country_exact.append(1.0 if (s1_ctry == c_ctry and s1_ctry != "") else 0.0)
        block_count.append(float(len(bt.split(",")) if bt else 1.0))

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
    print("=" * 65)
    print("SCORING CANDIDATES & BUILDING matching_results.tsv")
    print(f"Decision Threshold: {THRESHOLD}")
    print("=" * 65)

    print("\n[1/4] Loading model and slim source metadata...")
    booster = lgb.Booster(model_file=str(MODEL_DIR / "lgbm_matcher.txt"))

    s1_lookup = pl.read_parquet(CANDIDATE_DIR / "slim_parquet" / "s1.parquet").select([
        pl.col("entity_id").alias("source1_entity_id"),
        pl.col("normalized_name").alias("s1_name"),
        pl.col("normalized_name_core").alias("s1_name_core"),
        pl.col("normalized_address").alias("s1_address"),
        pl.col("normalized_country").alias("s1_country"),
    ])

    s2 = pl.read_parquet(CANDIDATE_DIR / "slim_parquet" / "s2.parquet").select([
        pl.col("entity_id").alias("candidate_entity_id"),
        pl.col("normalized_name").alias("match_name"),
        pl.col("normalized_name_core").alias("match_name_core"),
        pl.col("normalized_address").alias("match_address"),
        pl.col("normalized_country").alias("match_country"),
    ])

    s3 = pl.read_parquet(CANDIDATE_DIR / "slim_parquet" / "s3.parquet").select([
        pl.col("entity_id").alias("candidate_entity_id"),
        pl.col("normalized_name").alias("match_name"),
        pl.col("normalized_name_core").alias("match_name_core"),
        pl.col("normalized_address").alias("match_address"),
        pl.col("normalized_country").alias("match_country"),
    ])

    other_lookup = pl.concat([s2, s3], how="vertical")
    del s2, s3
    gc.collect()

    partition_files = sorted((CANDIDATE_DIR / "candidate_partitions").glob("candidate_part_*.parquet"))
    print(f"\n[2/4] Scoring {len(partition_files)} partitions...")

    temp_part_files = []

    for idx, p_file in enumerate(partition_files, 1):
        temp_out = TEMP_PARTS_DIR / f"scored_part_{idx:03d}.parquet"
        temp_part_files.append(temp_out)

        if temp_out.exists():
            print(f"  [reuse] [{idx}/{len(partition_files)}] {temp_out.name}")
            continue

        print(f"  [processing] [{idx}/{len(partition_files)}] {p_file.name}...")

        df = (
            pl.read_parquet(p_file)
            .join(s1_lookup, on="source1_entity_id", how="left")
            .join(other_lookup, on="candidate_entity_id", how="left")
        )

        df = compute_features(df)
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        X = df.select(feature_cols).to_numpy()

        probs = booster.predict(X)

        matches_df = (
            df.select(["source1_entity_id", "candidate_entity_id"])
            .with_columns(pl.Series("prob", probs, dtype=pl.Float32))
            .filter(pl.col("prob") >= THRESHOLD)
            .select(["source1_entity_id", "candidate_entity_id"])
        )

        matches_df.write_parquet(temp_out)
        del df, X, probs, matches_df
        gc.collect()

    print("\n[3/4] Aggregating matched candidates per Source 1 entity...")
    matches_lf = pl.concat([pl.scan_parquet(p) for p in temp_part_files], how="vertical")

    grouped_matches = (
        matches_lf
        .group_by("source1_entity_id")
        .agg(pl.col("candidate_entity_id").unique().sort().alias("_ids"))
        .with_columns(pl.col("_ids").list.join(",").alias("matched_entity_ids"))
        .select(["source1_entity_id", "matched_entity_ids"])
    )

    print("\n[4/4] Writing final matching_results.tsv (ensuring all S1 IDs exist)...")
    s1_all = (
        pl.scan_parquet(CANDIDATE_DIR / "slim_parquet" / "s1.parquet")
        .select(pl.col("entity_id").alias("source1_entity_id"))
    )

    final_df = (
        s1_all
        .join(grouped_matches, on="source1_entity_id", how="left")
        .with_columns(pl.col("matched_entity_ids").fill_null(""))
        .select(["source1_entity_id", "matched_entity_ids"])
        .collect(engine="streaming")
    )

    final_df.write_csv(OUTPUT_FILE, separator="\t")

    print("=" * 65)
    print(f"SUCCESS: {OUTPUT_FILE} created!")
    print(f"Total Source 1 entities: {final_df.height:,}")
    print(f"Source 1 entities with matches: {final_df.filter(pl.col('matched_entity_ids') != '').height:,}")
    print("=" * 65)

if __name__ == "__main__":
    main()
