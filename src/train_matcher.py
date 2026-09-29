

from pathlib import Path
import gc
import re
import numpy as np
import polars as pl
import lightgbm as lgb
from rapidfuzz.distance import JaroWinkler, Levenshtein

CANDIDATE_DIR = Path("dataset/candidates")
NORMALIZED_DIR = Path("dataset/normalized")
TRAIN_DIR = Path("dataset/train")
MODEL_DIR = Path("dataset/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NUM_SAMPLE_PARTITIONS = 3  # Draw pool from 3 partitions
NEGATIVE_RATIO = 5         # 5 negatives per 1 positive (~1.5M negatives)

def extract_numbers(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"\b\d+\b", text))

def compute_features(df: pl.DataFrame) -> pl.DataFrame:
    """Compute vector/string similarity features on a batch of candidate pairs."""
    s1_names = df["s1_name"].to_list()
    c_names = df["match_name"].to_list()
    s1_cores = df["s1_name_core"].to_list()
    c_cores = df["match_name_core"].to_list()
    s1_addrs = df["s1_address"].to_list()
    c_addrs = df["match_address"].to_list()
    s1_countries = df["s1_country"].to_list()
    c_countries = df["match_country"].to_list()
    block_types = df["block_types"].to_list()

    jw_name = []
    jw_core = []
    lev_name = []
    tok_jaccard_name = []
    tok_jaccard_addr = []
    num_exact_addr = []
    country_exact = []
    block_count = []

    for s1_n, cn, s1_c, cc, s1_a, ca, s1_ctry, c_ctry, bt in zip(
        s1_names, c_names, s1_cores, c_cores, s1_addrs, c_addrs, s1_countries, c_countries, block_types
    ):
        s1_n = s1_n or ""
        cn = cn or ""
        s1_c = s1_c or ""
        cc = cc or ""
        s1_a = s1_a or ""
        ca = ca or ""

        # Name metrics
        jw_name.append(JaroWinkler.similarity(s1_n, cn))
        jw_core.append(JaroWinkler.similarity(s1_c, cc))
        lev_name.append(Levenshtein.normalized_similarity(s1_n, cn))

        # Token Jaccard
        t1_name, t2_name = set(s1_n.split()), set(cn.split())
        tok_jaccard_name.append(
            len(t1_name & t2_name) / len(t1_name | t2_name) if (t1_name or t2_name) else 0.0
        )

        t1_addr, t2_addr = set(s1_a.split()), set(ca.split())
        tok_jaccard_addr.append(
            len(t1_addr & t2_addr) / len(t1_addr | t2_addr) if (t1_addr or t2_addr) else 0.0
        )

        # Address number consistency
        n1_addr, n2_addr = extract_numbers(s1_a), extract_numbers(ca)
        if n1_addr and n2_addr:
            num_exact_addr.append(1.0 if len(n1_addr & n2_addr) > 0 else 0.0)
        else:
            num_exact_addr.append(0.5)

        # Metadata
        country_exact.append(1.0 if s1_ctry == c_ctry and s1_ctry != "" else 0.0)
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

def load_source_lookups():
    print("[1/4] Loading slim source lookups...")
    s1 = pl.read_parquet(CANDIDATE_DIR / "slim_parquet" / "s1.parquet").select([
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

    other = pl.concat([s2, s3], how="vertical")
    return s1, other

def load_ground_truth_pairs():
    print("[2/4] Loading ground truth labels...")
    gt = (
        pl.scan_csv(TRAIN_DIR / "train_ground_truth.tsv", separator="\t")
        .filter(pl.col("matched_entity_ids").fill_null("") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("_ids"))
        .explode("_ids")
        .with_columns(pl.col("_ids").str.strip_chars().alias("candidate_entity_id"))
        .select(["source1_entity_id", "candidate_entity_id"])
        .unique()
        .with_columns(pl.lit(1).alias("is_match"))
        .collect()
    )
    return gt

def main():
    s1_lookup, other_lookup = load_source_lookups()
    gt_pairs = load_ground_truth_pairs()

    print("[3/4] Preparing training sample from candidate partitions...")
    partition_files = sorted((CANDIDATE_DIR / "candidate_partitions").glob("candidate_part_*.parquet"))[:NUM_SAMPLE_PARTITIONS]

    sample_candidates = pl.concat([pl.read_parquet(p) for p in partition_files], how="vertical")

    # Join metadata
    df = (
        sample_candidates
        .join(s1_lookup, on="source1_entity_id", how="left")
        .join(other_lookup, on="candidate_entity_id", how="left")
        .join(gt_pairs, on=["source1_entity_id", "candidate_entity_id"], how="left")
        .with_columns(pl.col("is_match").fill_null(0))
    )

    del sample_candidates, s1_lookup, other_lookup, gt_pairs
    gc.collect()

    # Separate Positives and Negatives to downsample safely
    df_pos = df.filter(pl.col("is_match") == 1)
    df_neg = df.filter(pl.col("is_match") == 0)

    n_pos = df_pos.height
    n_neg_target = min(n_pos * NEGATIVE_RATIO, df_neg.height)

    print(f"Full partition pool: {df.height:,} rows (Positives: {n_pos:,}, Negatives: {df_neg.height:,})")
    print(f"Downsampling negatives to {n_neg_target:,} (ratio {NEGATIVE_RATIO}:1) for memory safety...")

    df_neg_sampled = df_neg.sample(n=n_neg_target, seed=42)
    
    del df, df_neg
    gc.collect()

    df_train_sample = pl.concat([df_pos, df_neg_sampled], how="vertical").sample(fraction=1.0, shuffle=True, seed=42)

    del df_pos, df_neg_sampled
    gc.collect()

    print(f"Effective training set: {df_train_sample.height:,} pairs")

    # Feature computation on the lightweight balanced sample
    print("Computing features...")
    df_feat = compute_features(df_train_sample)

    del df_train_sample
    gc.collect()

    feature_cols = [c for c in df_feat.columns if c.startswith("feat_")]
    X = df_feat.select(feature_cols).to_numpy()
    y = df_feat["is_match"].to_numpy()

    del df_feat
    gc.collect()

    # Train-val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print("[4/4] Training LightGBM classifier...")
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50)],
    )

    model_path = MODEL_DIR / "lgbm_matcher.txt"
    clf.booster_.save_model(str(model_path))
    print(f"\nModel trained and saved to {model_path}")

if __name__ == "__main__":
    main()
