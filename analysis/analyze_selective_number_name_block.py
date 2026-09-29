#!/usr/bin/env python3
"""
Shadow Experiment: Country + RARE Name Token + Address Number Blocking
Analyzes recovery against the 1,571,452 missed true pairs while pruning high-frequency tokens.
Diagnostic only. Does NOT modify production candidate partitions or build_candidates_5.py.
"""

from pathlib import Path
import os
import gc
import polars as pl

# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

NORMALIZED_DIR = Path("dataset/normalized")
GT_FILE = Path("dataset/train/train_ground_truth.tsv")
CANDIDATE_DIR = Path("dataset/candidates")
OUTPUT_DIR = Path("dataset/candidates/missed_analysis")
OUTPUT_FILE = OUTPUT_DIR / "selective_number_name_block_analysis.tsv"

NUM_PARTITIONS = 64
PAIR_CAP = 5000
SEP = "\x1f"
MIN_TOKEN_LEN = 3
RARE_TOKEN_MAX_FREQ = 2500

NAME_STOPWORDS = {
    "the", "and", "of", "for", "new", "inc", "llc", "ltd", "co", "company",
    "corporation", "corp", "group", "international", "intl", "private",
    "limited", "pvt", "enterprises", "enterprise", "sarl", "sas", "sa",
}

# =============================================================================
# HELPERS
# =============================================================================

def require_path(path: Path, desc: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required path ({desc}): {path}")

def detect_column(cols: list[str], candidates: list[str]) -> str:
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Could not find any of {candidates} in columns: {cols}")

# =============================================================================
# GROUND TRUTH & PRODUCTION CANDIDATES
# =============================================================================

def load_missed_true_pairs() -> pl.DataFrame:
    """
    Computes or reuses the exact set of missed true pairs partition-by-partition.
    """
    missed_parquet = OUTPUT_DIR / "missed_true_pairs.parquet"
    if missed_parquet.exists():
        print(f"[reuse] Missed true pairs from {missed_parquet}")
        df = pl.read_parquet(missed_parquet)
        s1_col = detect_column(df.columns, ["source1_entity_id", "s1_id"])
        c_col = detect_column(df.columns, ["candidate_entity_id", "candidate_id", "match_id", "entity_id"])
        return df.select([
            pl.col(s1_col).alias("s1_id"),
            pl.col(c_col).alias("candidate_id")
        ]).unique()

    print("[load] Extracting ground truth pairs...")
    require_path(GT_FILE, "Ground Truth")
    gt = (
        pl.scan_csv(GT_FILE, separator="\t")
        .filter(pl.col("matched_entity_ids").fill_null("") != "")
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("_ids"))
        .explode("_ids")
        .with_columns(pl.col("_ids").str.strip_chars().alias("candidate_id"))
        .select([
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("candidate_id")
        ])
        .unique()
        .collect()
    )

    print(f"Total true pairs: {gt.height:,}")
    print("[partition] Checking existing production partitions for misses...")

    found_parts = []
    for part in range(NUM_PARTITIONS):
        p_path = CANDIDATE_DIR / "candidate_partitions" / f"candidate_part_{part:03d}.parquet"
        require_path(p_path, f"Candidate Partition {part:03d}")
        
        c_lf = pl.scan_parquet(p_path)
        c_cols = c_lf.collect_schema().names()
        s1_c = detect_column(c_cols, ["source1_entity_id", "s1_id"])
        m_c = detect_column(c_cols, ["candidate_entity_id", "candidate_id", "match_id", "entity_id"])
        
        c_df = c_lf.select([
            pl.col(s1_c).alias("s1_id"),
            pl.col(m_c).alias("candidate_id")
        ]).unique().collect()

        gt_part = gt.filter(
            (pl.col("s1_id").hash().mod(NUM_PARTITIONS)) == part
        )
        
        found = gt_part.join(c_df, on=["s1_id", "candidate_id"], how="inner")
        if found.height > 0:
            found_parts.append(found)

    found_all = pl.concat(found_parts, how="vertical").unique()
    missed = gt.join(found_all, on=["s1_id", "candidate_id"], how="anti")
    print(f"Missed true pairs: {missed.height:,}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missed.write_parquet(missed_parquet)
    return missed

# =============================================================================
# FEATURE EXTRACTION & RARE TOKEN SELECTION
# =============================================================================

def prepare_rare_token_records(source_file: Path, max_token_freq: int) -> pl.DataFrame:
    """
    Extracts address number and selects only rare name-core tokens within country.
    """
    require_path(source_file, f"Normalized Source: {source_file.name}")
    print(f"[scan] Loading {source_file.name}...")

    df = (
        pl.read_csv(source_file, separator="\t")
        .select([
            pl.col("entity_id"),
            pl.col("normalized_country").fill_null(""),
            pl.col("normalized_name_core").fill_null(""),
            pl.col("normalized_address").fill_null(""),
        ])
        .with_columns(
            pl.col("normalized_address")
            .str.extract(r"(?:^|[^0-9])([0-9]{1,6})(?:[^0-9]|$)", group_index=1)
            .fill_null("")
            .alias("address_num")
        )
        .filter(
            (pl.col("normalized_country") != "")
            & (pl.col("address_num") != "")
            & (pl.col("normalized_name_core") != "")
        )
    )

    # Explode tokens
    tok_df = (
        df.select([
            pl.col("entity_id"),
            pl.col("normalized_country"),
            pl.col("address_num"),
            pl.col("normalized_name_core").str.split(" ").alias("tok")
        ])
        .explode("tok")
        .with_columns(pl.col("tok").str.strip_chars().alias("tok"))
        .filter(
            (pl.col("tok").str.len_chars() >= MIN_TOKEN_LEN)
            & (~pl.col("tok").is_in(list(NAME_STOPWORDS)))
        )
        .unique(subset=["entity_id", "normalized_country", "tok"])
    )

    # Compute token frequency within country
    print(f"[freq] Computing token frequencies for {source_file.name}...")
    freq_df = (
        tok_df.group_by(["normalized_country", "tok"])
        .agg(pl.col("entity_id").n_unique().alias("freq"))
        .filter(pl.col("freq") <= max_token_freq)
    )

    # Filter to rare tokens only
    rare_tok_df = (
        tok_df.join(freq_df.select(["normalized_country", "tok"]), on=["normalized_country", "tok"], how="inner")
        .with_columns(
            (
                pl.col("normalized_country")
                + pl.lit(SEP)
                + pl.col("tok")
                + pl.lit(SEP)
                + pl.col("address_num")
            ).alias("block_key")
        )
        .select(["entity_id", "block_key"])
        .unique()
    )

    print(f"[{source_file.stem}] Generated {rare_tok_df.height:,} rare-token records.")
    return rare_tok_df

# =============================================================================
# BLOCKING EVALUATION
# =============================================================================

def analyze_selective_block():
    print("=" * 70)
    print("SHADOW EXPERIMENT: Country + RARE Name Token (<=200) + Address Number")
    print("=" * 70)

    missed_df = load_missed_true_pairs()
    total_missed = missed_df.height

    s1_rare = prepare_rare_token_records(NORMALIZED_DIR / "train_source1.tsv", RARE_TOKEN_MAX_FREQ)
    s2_rare = prepare_rare_token_records(NORMALIZED_DIR / "train_source2.tsv", RARE_TOKEN_MAX_FREQ)
    s3_rare = prepare_rare_token_records(NORMALIZED_DIR / "train_source3.tsv", RARE_TOKEN_MAX_FREQ)

    stats = {}

    for src_label, other_df in [("S2", s2_rare), ("S3", s3_rare)]:
        print(f"\n--- Analyzing S1 x {src_label} ---")
        
        s1_counts = s1_rare.group_by("block_key").agg(pl.col("entity_id").n_unique().alias("n_s1"))
        other_counts = other_df.group_by("block_key").agg(pl.col("entity_id").n_unique().alias("n_other"))

        block_sizes = (
            s1_counts.join(other_counts, on="block_key", how="inner")
            .with_columns((pl.col("n_s1") * pl.col("n_other")).alias("pair_count"))
        )

        largest_block = block_sizes.select(pl.col("pair_count").max()).item() or 0
        overflow_blocks = block_sizes.filter(pl.col("pair_count") > PAIR_CAP).height

        # Safe blocks under cap
        safe_blocks = block_sizes.filter(pl.col("pair_count") <= PAIR_CAP)
        cand_pairs = safe_blocks.select(pl.col("pair_count").sum()).item() or 0

        print(f"[{src_label}] Valid cross-blocks: {block_sizes.height:,}")
        print(f"[{src_label}] Largest block: {largest_block:,}")
        print(f"[{src_label}] Overflow blocks (> {PAIR_CAP}): {overflow_blocks:,}")
        print(f"[{src_label}] Estimated candidates (capped): {cand_pairs:,}")

        # Materialize safe pairs for recovery evaluation against missed set
        s1_sub = s1_rare.join(safe_blocks.select("block_key"), on="block_key", how="inner")
        other_sub = other_df.join(safe_blocks.select("block_key"), on="block_key", how="inner")

        matched_pairs = (
            s1_sub.join(other_sub, on="block_key", how="inner")
            .select([
                pl.col("entity_id").alias("s1_id"),
                pl.col("entity_id_right").alias("candidate_id")
            ])
            .unique()
        )

        stats[src_label] = {
            "cand_pairs": cand_pairs,
            "largest_block": largest_block,
            "overflow_blocks": overflow_blocks,
            "matched_pairs": matched_pairs
        }

        del s1_sub, other_sub
        gc.collect()

    print("\n--- Evaluating Recovery Against Missed True Pairs ---")
    all_recovered_pairs = pl.concat([
        stats["S2"]["matched_pairs"],
        stats["S3"]["matched_pairs"]
    ], how="vertical").unique()

    recovered_df = missed_df.join(all_recovered_pairs, on=["s1_id", "candidate_id"], how="inner")
    recovered_count = recovered_df.height
    recovery_pct = (recovered_count / total_missed * 100.0) if total_missed else 0.0

    total_candidates = stats["S2"]["cand_pairs"] + stats["S3"]["cand_pairs"]
    cand_per_recovery = (total_candidates / recovered_count) if recovered_count > 0 else float("inf")

    # =========================================================================
    # PRINT RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"Currently missed true pairs   : {total_missed:,}")
    print(f"Recovered true pairs          : {recovered_count:,}")
    print(f"Recovery percentage           : {recovery_pct:.4f}%")
    print(f"Estimated candidate pairs     : {total_candidates:,}")
    print(f"Candidates per recovered pair : {cand_per_recovery:.2f}")
    print(f"S2 candidate count            : {stats['S2']['cand_pairs']:,}")
    print(f"S3 candidate count            : {stats['S3']['cand_pairs']:,}")
    print(f"S2 largest block              : {stats['S2']['largest_block']:,}")
    print(f"S3 largest block              : {stats['S3']['largest_block']:,}")
    print(f"S2 overflow blocks            : {stats['S2']['overflow_blocks']:,}")
    print(f"S3 overflow blocks            : {stats['S3']['overflow_blocks']:,}")
    print("=" * 70)

    # Save summary TSV
    summary_data = [
        {"metric": "currently_missed_true_pairs", "value": str(total_missed)},
        {"metric": "recovered_true_pairs", "value": str(recovered_count)},
        {"metric": "recovery_percentage", "value": f"{recovery_pct:.4f}%"},
        {"metric": "estimated_candidate_pairs", "value": str(total_candidates)},
        {"metric": "candidates_per_recovered_pair", "value": f"{cand_per_recovery:.2f}"},
        {"metric": "s2_candidate_count", "value": str(stats['S2']['cand_pairs'])},
        {"metric": "s3_candidate_count", "value": str(stats['S3']['cand_pairs'])},
        {"metric": "s2_largest_block", "value": str(stats['S2']['largest_block'])},
        {"metric": "s3_largest_block", "value": str(stats['S3']['largest_block'])},
        {"metric": "s2_overflow_blocks", "value": str(stats['S2']['overflow_blocks'])},
        {"metric": "s3_overflow_blocks", "value": str(stats['S3']['overflow_blocks'])},
    ]

    pl.DataFrame(summary_data).write_csv(OUTPUT_FILE, separator="\t")
    print(f"\n[saved] Results written to: {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_selective_block()
