#!/usr/bin/env python3

"""
Analyze whether exact-name / name-core true-match pairs were missed
because of the MAX_PAIRS_PER_BLOCK_KEY cap.

This script is designed for the current entity-resolution pipeline.

It analyzes:
    1. Exact normalized-name matches that were missed.
    2. Exact normalized-name-core matches that were missed.
    3. The number of S1/S2/S3 records sharing each key.
    4. The theoretical Cartesian-product size for the block.
    5. Whether that theoretical block exceeds MAX_PAIRS_PER_BLOCK_KEY.
    6. Whether the missed pair exists in the corresponding raw block file.
    7. Whether the evidence is consistent with cap-related truncation.

Important:
    - Raw data is never modified.
    - Existing candidate/block files are never modified.
    - This is diagnostic only.
    - It does NOT automatically change the blocking strategy.

Usage:

    python3 analyze_block_cap_misses.py

Optional:

    python3 analyze_block_cap_misses.py \
        --train-dir dataset/train \
        --normalized-dir dataset/normalized \
        --candidate-dir dataset/candidates \
        --output-dir dataset/candidates/missed_analysis \
        --cap 5000

"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import polars as pl


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_TRAIN_DIR = "dataset/train"
DEFAULT_NORMALIZED_DIR = "dataset/normalized"
DEFAULT_CANDIDATE_DIR = "dataset/candidates"
DEFAULT_OUTPUT_DIR = "dataset/candidates/missed_analysis"

DEFAULT_CAP = 5000


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(message, flush=True)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {description}:\n"
            f"  {path}\n"
        )


def get_source_columns(df: pl.DataFrame) -> list[str]:
    required = [
        "entity_id",
        "normalized_name",
        "normalized_name_core",
        "normalized_address",
        "normalized_country",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required normalized columns: {missing}\n"
            f"Available columns: {df.columns}"
        )

    return required


def read_normalized_source(path: Path) -> pl.DataFrame:
    log(f"Reading normalized source: {path}")

    df = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10000,
        null_values=[""],
        ignore_errors=False,
    )

    get_source_columns(df)

    return df.select(
        [
            "entity_id",
            "normalized_name",
            "normalized_name_core",
            "normalized_address",
            "normalized_country",
        ]
    )


def detect_source(entity_id: str) -> str:
    if entity_id.startswith("S1-"):
        return "S1"
    if entity_id.startswith("S2-"):
        return "S2"
    if entity_id.startswith("S3-"):
        return "S3"

    raise ValueError(f"Unknown entity source ID: {entity_id}")


# ============================================================
# LOAD MISSED TRUE PAIRS
# ============================================================

def load_missed_pairs(candidate_dir: Path) -> pl.DataFrame:
    missed_path = (
        candidate_dir
        / "missed_analysis"
        / "missed_true_pairs.parquet"
    )

    require_file(
        missed_path,
        "missed true-pairs file",
    )

    log("")
    log("=" * 70)
    log("LOADING MISSED TRUE PAIRS")
    log("=" * 70)
    log(f"File: {missed_path}")

    df = pl.read_parquet(missed_path)

    log(f"Rows: {df.height:,}")
    log(f"Columns: {df.columns}")

    # The analyzer created earlier may contain these columns.
    # We only require the pair IDs.
    possible_s1 = [
        "source1_entity_id",
        "s1_entity_id",
        "entity_id_s1",
        "s1_id",
    ]

    possible_match = [
        "matched_entity_id",
        "match_entity_id",
        "candidate_entity_id",
        "entity_id_match",
        "matched_id",
    ]

    s1_col = next(
        (c for c in possible_s1 if c in df.columns),
        None,
    )

    match_col = next(
        (c for c in possible_match if c in df.columns),
        None,
    )

    if s1_col is None or match_col is None:
        raise ValueError(
            "\nCould not identify the S1/matched-ID columns in "
            "missed_true_pairs.parquet.\n\n"
            f"Available columns:\n{df.columns}\n\n"
            "Expected one of:\n"
            f"S1 columns: {possible_s1}\n"
            f"Match columns: {possible_match}\n"
        )

    log(f"S1 ID column:     {s1_col}")
    log(f"Matched ID column: {match_col}")

    df = (
        df.select(
            [
                pl.col(s1_col).cast(pl.Utf8).alias("s1_id"),
                pl.col(match_col).cast(pl.Utf8).alias("match_id"),
            ]
        )
        .filter(
            pl.col("s1_id").is_not_null()
            & pl.col("match_id").is_not_null()
        )
        .unique()
    )

    # Keep only genuine S1 -> S2/S3 pairs.
    df = df.filter(
        pl.col("s1_id").str.starts_with("S1-")
        & (
            pl.col("match_id").str.starts_with("S2-")
            | pl.col("match_id").str.starts_with("S3-")
        )
    )

    log(f"Valid missed pairs: {df.height:,}")

    return df


# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

def load_normalized_data(normalized_dir: Path) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    log("")
    log("=" * 70)
    log("LOADING NORMALIZED DATA")
    log("=" * 70)

    paths = {
        "S1": normalized_dir / "train_source1.tsv",
        "S2": normalized_dir / "train_source2.tsv",
        "S3": normalized_dir / "train_source3.tsv",
    }

    for source, path in paths.items():
        require_file(path, f"{source} normalized file")

    s1 = read_normalized_source(paths["S1"])
    s2 = read_normalized_source(paths["S2"])
    s3 = read_normalized_source(paths["S3"])

    log("")
    log(f"S1 rows: {s1.height:,}")
    log(f"S2 rows: {s2.height:,}")
    log(f"S3 rows: {s3.height:,}")

    return s1, s2, s3


# ============================================================
# BUILD KEY FREQUENCY TABLES
# ============================================================

def build_key_counts(
    df: pl.DataFrame,
    source_name: str,
    key_column: str,
) -> pl.DataFrame:

    log(
        f"Building {source_name} "
        f"{key_column} frequency table..."
    )

    result = (
        df.filter(
            pl.col(key_column).is_not_null()
            & (pl.col(key_column) != "")
        )
        .group_by(key_column)
        .agg(
            pl.len().alias(f"{source_name.lower()}_count")
        )
    )

    return result


def build_all_key_counts(
    s1: pl.DataFrame,
    s2: pl.DataFrame,
    s3: pl.DataFrame,
    key_column: str,
) -> pl.DataFrame:

    s1_counts = build_key_counts(s1, "S1", key_column)
    s2_counts = build_key_counts(s2, "S2", key_column)
    s3_counts = build_key_counts(s3, "S3", key_column)

    result = (
        s1_counts
        .join(
            s2_counts,
            on=key_column,
            how="full",
            coalesce=True,
        )
        .join(
            s3_counts,
            on=key_column,
            how="full",
            coalesce=True,
        )
        .with_columns(
            [
                pl.col("s1_count").fill_null(0),
                pl.col("s2_count").fill_null(0),
                pl.col("s3_count").fill_null(0),
            ]
        )
        .with_columns(
            [
                (
                    pl.col("s1_count") * pl.col("s2_count")
                ).alias("s1_s2_pairs"),

                (
                    pl.col("s1_count") * pl.col("s3_count")
                ).alias("s1_s3_pairs"),
            ]
        )
        .with_columns(
            (
                pl.col("s1_s2_pairs")
                + pl.col("s1_s3_pairs")
            ).alias("total_cross_source_pairs")
        )
    )

    return result


# ============================================================
# ADD RECORD-LEVEL INFORMATION
# ============================================================

def prepare_sources(
    s1: pl.DataFrame,
    s2: pl.DataFrame,
    s3: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:

    s1_small = s1.select(
        [
            pl.col("entity_id").alias("s1_id"),
            pl.col("normalized_name").alias("s1_name"),
            pl.col("normalized_name_core").alias("s1_name_core"),
            pl.col("normalized_address").alias("s1_address"),
            pl.col("normalized_country").alias("s1_country"),
        ]
    )

    s2_small = s2.select(
        [
            pl.col("entity_id").alias("match_id"),
            pl.col("normalized_name").alias("match_name"),
            pl.col("normalized_name_core").alias("match_name_core"),
            pl.col("normalized_address").alias("match_address"),
            pl.col("normalized_country").alias("match_country"),
        ]
    )

    s3_small = s3.select(
        [
            pl.col("entity_id").alias("match_id"),
            pl.col("normalized_name").alias("match_name"),
            pl.col("normalized_name_core").alias("match_name_core"),
            pl.col("normalized_address").alias("match_address"),
            pl.col("normalized_country").alias("match_country"),
        ]
    )

    matches = pl.concat(
        [s2_small, s3_small],
        how="vertical",
    )

    return s1_small, matches


# ============================================================
# JOIN MISSED PAIRS TO RECORD DATA
# ============================================================

def attach_record_data(
    missed: pl.DataFrame,
    s1: pl.DataFrame,
    matches: pl.DataFrame,
) -> pl.DataFrame:

    log("")
    log("=" * 70)
    log("ATTACHING RECORD INFORMATION")
    log("=" * 70)

    result = (
        missed
        .join(
            s1,
            on="s1_id",
            how="left",
        )
        .join(
            matches,
            on="match_id",
            how="left",
        )
    )

    # Identify source of matched entity.
    result = result.with_columns(
        pl.when(
            pl.col("match_id").str.starts_with("S2-")
        )
        .then(pl.lit("S2"))
        .when(
            pl.col("match_id").str.starts_with("S3-")
        )
        .then(pl.lit("S3"))
        .otherwise(pl.lit(""))
        .alias("match_source")
    )

    return result


# ============================================================
# ATTACH NAME BLOCK STATISTICS
# ============================================================

def attach_block_statistics(
    df: pl.DataFrame,
    name_counts: pl.DataFrame,
    core_counts: pl.DataFrame,
    cap: int,
) -> pl.DataFrame:

    log("")
    log("=" * 70)
    log("ATTACHING BLOCK SIZE INFORMATION")
    log("=" * 70)

    # --------------------------------------------------------
    # Exact normalized-name block
    # --------------------------------------------------------

    name_counts = name_counts.rename(
        {
            "s1_count": "name_s1_count",
            "s2_count": "name_s2_count",
            "s3_count": "name_s3_count",
            "s1_s2_pairs": "name_s1_s2_pairs",
            "s1_s3_pairs": "name_s1_s3_pairs",
            "total_cross_source_pairs": "name_total_pairs",
        }
    )

    df = df.join(
        name_counts,
        left_on="s1_name",
        right_on="normalized_name",
        how="left",
    )

    # --------------------------------------------------------
    # Name-core block
    # --------------------------------------------------------

    core_counts = core_counts.rename(
        {
            "s1_count": "core_s1_count",
            "s2_count": "core_s2_count",
            "s3_count": "core_s3_count",
            "s1_s2_pairs": "core_s1_s2_pairs",
            "s1_s3_pairs": "core_s1_s3_pairs",
            "total_cross_source_pairs": "core_total_pairs",
        }
    )

    df = df.join(
        core_counts,
        left_on="s1_name_core",
        right_on="normalized_name_core",
        how="left",
    )

    # --------------------------------------------------------
    # Fill missing statistics
    # --------------------------------------------------------

    numeric_columns = [
        "name_s1_count",
        "name_s2_count",
        "name_s3_count",
        "name_s1_s2_pairs",
        "name_s1_s3_pairs",
        "name_total_pairs",
        "core_s1_count",
        "core_s2_count",
        "core_s3_count",
        "core_s1_s2_pairs",
        "core_s1_s3_pairs",
        "core_total_pairs",
    ]

    df = df.with_columns(
        [
            pl.col(c).fill_null(0)
            for c in numeric_columns
        ]
    )

    # --------------------------------------------------------
    # Which exact block applies to this pair?
    # --------------------------------------------------------

    df = df.with_columns(
        [
            (
                (pl.col("s1_name") != "")
                & (pl.col("s1_name") == pl.col("match_name"))
            ).alias("name_exact_match"),

            (
                (pl.col("s1_name_core") != "")
                & (
                    pl.col("s1_name_core")
                    == pl.col("match_name_core")
                )
            ).alias("name_core_exact_match"),
        ]
    )

    # --------------------------------------------------------
    # Theoretical cap risk
    # --------------------------------------------------------

    df = df.with_columns(
        [
            (
                pl.col("name_total_pairs") > cap
            ).alias("name_block_over_cap"),

            (
                pl.col("core_total_pairs") > cap
            ).alias("name_core_block_over_cap"),
        ]
    )

    # --------------------------------------------------------
    # If the pair belongs to an exact block, determine whether
    # that exact block would be larger than the cap.
    # --------------------------------------------------------

    df = df.with_columns(
        [
            (
                pl.col("name_exact_match")
                & pl.col("name_block_over_cap")
            ).alias("name_exact_cap_risk"),

            (
                pl.col("name_core_exact_match")
                & pl.col("name_core_block_over_cap")
            ).alias("name_core_cap_risk"),
        ]
    )

    # --------------------------------------------------------
    # Combined interpretation
    # --------------------------------------------------------

    df = df.with_columns(
        [
            (
                pl.col("name_exact_cap_risk")
                | pl.col("name_core_cap_risk")
            ).alias("possible_cap_miss")
        ]
    )

    return df


# ============================================================
# CHECK WHETHER MISSED PAIRS EXIST IN EXISTING BLOCK FILES
# ============================================================

def check_existing_block(
    missed_df: pl.DataFrame,
    block_path: Path,
    label: str,
) -> pl.DataFrame:

    if not block_path.exists():
        log(f"Block file not found, skipping: {block_path}")
        return missed_df.with_columns(
            pl.lit(False).alias(f"in_{label}")
        )

    log(f"Checking existing block: {block_path}")

    block = pl.read_parquet(block_path)

    # --------------------------------------------------------
    # Detect ID columns
    # --------------------------------------------------------

    cols = block.columns

    s1_candidates = [
        "source1_entity_id",
        "s1_id",
        "entity_id_s1",
        "source1_id",
    ]

    match_candidates = [
        "matched_entity_id",
        "match_id",
        "candidate_entity_id",
        "entity_id_s2",
        "entity_id_s3",
        "source2_entity_id",
        "source3_entity_id",
    ]

    s1_col = next(
        (c for c in s1_candidates if c in cols),
        None,
    )

    match_col = next(
        (c for c in match_candidates if c in cols),
        None,
    )

    # If standard pair columns are unavailable, inspect first
    # two columns that look like IDs.
    if s1_col is None:
        for c in cols:
            if c.lower() in {"s1", "source1"}:
                s1_col = c
                break

    if match_col is None:
        for c in cols:
            if c != s1_col and (
                "entity" in c.lower()
                or "match" in c.lower()
                or "candidate" in c.lower()
            ):
                match_col = c
                break

    if s1_col is None or match_col is None:
        log(
            f"WARNING: Could not identify pair columns in {block_path}"
        )
        log(f"Columns: {cols}")

        return missed_df.with_columns(
            pl.lit(False).alias(f"in_{label}")
        )

    block_pairs = (
        block.select(
            [
                pl.col(s1_col)
                .cast(pl.Utf8)
                .alias("s1_id"),

                pl.col(match_col)
                .cast(pl.Utf8)
                .alias("match_id"),
            ]
        )
        .unique()
        .with_columns(
            pl.lit(True).alias(f"in_{label}")
        )
    )

    result = missed_df.join(
        block_pairs,
        on=["s1_id", "match_id"],
        how="left",
    )

    return result.with_columns(
        pl.col(f"in_{label}").fill_null(False)
    )


def check_all_existing_blocks(
    df: pl.DataFrame,
    candidate_dir: Path,
) -> pl.DataFrame:

    log("")
    log("=" * 70)
    log("CHECKING EXISTING EXACT BLOCK FILES")
    log("=" * 70)

    block_paths = {
        "name_exact": candidate_dir / "S2_name_exact.parquet",
        "name_core": candidate_dir / "S2_name_core.parquet",
        "address_exact": candidate_dir / "S2_address_exact.parquet",
    }

    # S3 files are checked separately because the same S1 can
    # match either source.

    for label, path in block_paths.items():
        df = check_existing_block(
            df,
            path,
            label,
        )

    s3_block_paths = {
        "name_exact_s3": candidate_dir / "S3_name_exact.parquet",
        "name_core_s3": candidate_dir / "S3_name_core.parquet",
        "address_exact_s3": candidate_dir / "S3_address_exact.parquet",
    }

    for label, path in s3_block_paths.items():
        df = check_existing_block(
            df,
            path,
            label,
        )

    return df


# ============================================================
# CLASSIFY EACH MISSED PAIR
# ============================================================

def classify(df: pl.DataFrame, cap: int) -> pl.DataFrame:

    log("")
    log("=" * 70)
    log("CLASSIFYING MISSED PAIRS")
    log("=" * 70)

    df = df.with_columns(
        [
            (
                pl.col("name_exact_match")
                & ~pl.col("in_name_exact")
                & ~pl.col("in_name_exact_s3")
            ).alias("exact_name_missing_from_block"),

            (
                pl.col("name_core_exact_match")
                & ~pl.col("in_name_core")
                & ~pl.col("in_name_core_s3")
            ).alias("exact_core_missing_from_block"),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("name_exact_cap_risk")
                & pl.col("exact_name_missing_from_block")
            ).alias("strong_name_cap_evidence"),

            (
                pl.col("name_core_cap_risk")
                & pl.col("exact_core_missing_from_block")
            ).alias("strong_core_cap_evidence"),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("strong_name_cap_evidence")
                | pl.col("strong_core_cap_evidence")
            ).alias("strong_cap_evidence")
        ]
    )

    # --------------------------------------------------------
    # More descriptive category
    # --------------------------------------------------------

    df = df.with_columns(
        pl.when(
            pl.col("strong_name_cap_evidence")
        )
        .then(pl.lit("exact_name_block_over_cap_and_pair_absent"))

        .when(
            pl.col("strong_core_cap_evidence")
        )
        .then(pl.lit("name_core_block_over_cap_and_pair_absent"))

        .when(
            pl.col("name_exact_match")
            & (
                pl.col("in_name_exact")
                | pl.col("in_name_exact_s3")
            )
        )
        .then(pl.lit("exact_name_pair_present_in_block"))

        .when(
            pl.col("name_core_exact_match")
            & (
                pl.col("in_name_core")
                | pl.col("in_name_core_s3")
            )
        )
        .then(pl.lit("name_core_pair_present_in_block"))

        .when(
            pl.col("name_exact_match")
        )
        .then(pl.lit("exact_name_missed_not_explained_by_cap"))

        .when(
            pl.col("name_core_exact_match")
        )
        .then(pl.lit("name_core_missed_not_explained_by_cap"))

        .otherwise(
            pl.lit("not_exact_name_or_core")
        )
        .alias("cap_analysis_category")
    )

    return df


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    df: pl.DataFrame,
    output_dir: Path,
    cap: int,
) -> None:

    summary_rows = []

    total = df.height

    def add(metric: str, count: int):
        percentage = (
            (count / total * 100.0)
            if total
            else 0.0
        )

        summary_rows.append(
            {
                "metric": metric,
                "count": count,
                "percentage": percentage,
            }
        )

    add(
        "total_missed_pairs_analyzed",
        total,
    )

    add(
        "exact_name_true_matches",
        int(df["name_exact_match"].sum()),
    )

    add(
        "name_core_exact_true_matches",
        int(df["name_core_exact_match"].sum()),
    )

    add(
        "name_exact_block_over_cap",
        int(df["name_block_over_cap"].sum()),
    )

    add(
        "name_core_block_over_cap",
        int(df["name_core_block_over_cap"].sum()),
    )

    add(
        "exact_name_cap_risk",
        int(df["name_exact_cap_risk"].sum()),
    )

    add(
        "name_core_cap_risk",
        int(df["name_core_cap_risk"].sum()),
    )

    add(
        "strong_name_cap_evidence",
        int(df["strong_name_cap_evidence"].sum()),
    )

    add(
        "strong_core_cap_evidence",
        int(df["strong_core_cap_evidence"].sum()),
    )

    add(
        "strong_cap_evidence_any",
        int(df["strong_cap_evidence"].sum()),
    )

    add(
        "exact_name_missing_from_existing_block",
        int(df["exact_name_missing_from_block"].sum()),
    )

    add(
        "exact_core_missing_from_existing_block",
        int(df["exact_core_missing_from_block"].sum()),
    )

    summary = pl.DataFrame(summary_rows)

    summary.write_csv(
        output_dir / "block_cap_summary.tsv",
        separator="\t",
    )

    # Category summary
    category_summary = (
        df.group_by("cap_analysis_category")
        .agg(
            pl.len().alias("count")
        )
        .with_columns(
            (
                pl.col("count")
                / total
                * 100.0
            ).alias("percentage")
            if total
            else pl.lit(0.0).alias("percentage")
        )
        .sort("count", descending=True)
    )

    category_summary.write_csv(
        output_dir / "block_cap_category_summary.tsv",
        separator="\t",
    )

    log("")
    log("=" * 70)
    log("BLOCK CAP SUMMARY")
    log("=" * 70)

    print(summary)

    log("")
    log("CATEGORY SUMMARY")
    log("=" * 70)

    print(category_summary)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze whether missed exact-name/name-core "
            "matches were caused by block-size capping."
        )
    )

    parser.add_argument(
        "--train-dir",
        default=DEFAULT_TRAIN_DIR,
    )

    parser.add_argument(
        "--normalized-dir",
        default=DEFAULT_NORMALIZED_DIR,
    )

    parser.add_argument(
        "--candidate-dir",
        default=DEFAULT_CANDIDATE_DIR,
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CAP,
        help="MAX_PAIRS_PER_BLOCK_KEY used during blocking.",
    )

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    normalized_dir = Path(args.normalized_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("")
    log("=" * 70)
    log("BLOCK CAP MISS ANALYSIS")
    log("=" * 70)
    log(f"Train directory:      {train_dir}")
    log(f"Normalized directory: {normalized_dir}")
    log(f"Candidate directory:  {candidate_dir}")
    log(f"Output directory:     {output_dir}")
    log(f"Block cap:            {args.cap:,}")
    log("=" * 70)

    # --------------------------------------------------------
    # 1. Load missed true pairs
    # --------------------------------------------------------

    missed = load_missed_pairs(candidate_dir)

    if missed.height == 0:
        log("")
        log("No missed pairs found.")
        return

    # --------------------------------------------------------
    # 2. Load normalized data
    # --------------------------------------------------------

    s1, s2, s3 = load_normalized_data(
        normalized_dir
    )

    # --------------------------------------------------------
    # 3. Prepare source tables
    # --------------------------------------------------------

    s1_small, matches = prepare_sources(
        s1,
        s2,
        s3,
    )

    # --------------------------------------------------------
    # 4. Attach record-level data
    # --------------------------------------------------------

    df = attach_record_data(
        missed,
        s1_small,
        matches,
    )

    # --------------------------------------------------------
    # 5. Build exact-name frequency table
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("BUILDING EXACT-NAME KEY COUNTS")
    log("=" * 70)

    name_counts = build_all_key_counts(
        s1,
        s2,
        s3,
        "normalized_name",
    )

    # --------------------------------------------------------
    # 6. Build name-core frequency table
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("BUILDING NAME-CORE KEY COUNTS")
    log("=" * 70)

    core_counts = build_all_key_counts(
        s1,
        s2,
        s3,
        "normalized_name_core",
    )

    # --------------------------------------------------------
    # 7. Attach block statistics
    # --------------------------------------------------------

    df = attach_block_statistics(
        df,
        name_counts,
        core_counts,
        args.cap,
    )

    # --------------------------------------------------------
    # 8. Check whether pairs exist in existing block files
    # --------------------------------------------------------

    df = check_all_existing_blocks(
        df,
        candidate_dir,
    )

    # --------------------------------------------------------
    # 9. Classify
    # --------------------------------------------------------

    df = classify(
        df,
        args.cap,
    )

    # --------------------------------------------------------
    # 10. Save detailed diagnostics
    # --------------------------------------------------------

    detailed_path = (
        output_dir
        / "block_cap_diagnostics.parquet"
    )

    df.write_parquet(
        detailed_path,
        compression="zstd",
    )

    log("")
    log(f"Detailed diagnostics written to:")
    log(f"  {detailed_path}")

    # --------------------------------------------------------
    # 11. Save human-readable TSV
    # --------------------------------------------------------

    tsv_columns = [
        "s1_id",
        "match_id",
        "match_source",

        "s1_name",
        "match_name",

        "s1_name_core",
        "match_name_core",

        "s1_address",
        "match_address",

        "s1_country",
        "match_country",

        "name_exact_match",
        "name_core_exact_match",

        "name_s1_count",
        "name_s2_count",
        "name_s3_count",
        "name_s1_s2_pairs",
        "name_s1_s3_pairs",
        "name_total_pairs",

        "core_s1_count",
        "core_s2_count",
        "core_s3_count",
        "core_s1_s2_pairs",
        "core_s1_s3_pairs",
        "core_total_pairs",

        "name_block_over_cap",
        "name_core_block_over_cap",

        "name_exact_cap_risk",
        "name_core_cap_risk",

        "in_name_exact",
        "in_name_exact_s3",

        "in_name_core",
        "in_name_core_s3",

        "exact_name_missing_from_block",
        "exact_core_missing_from_block",

        "strong_name_cap_evidence",
        "strong_core_cap_evidence",
        "strong_cap_evidence",

        "cap_analysis_category",
    ]

    existing_columns = [
        c for c in tsv_columns
        if c in df.columns
    ]

    diagnostics_tsv = (
        output_dir
        / "block_cap_diagnostics.tsv"
    )

    df.select(existing_columns).write_csv(
        diagnostics_tsv,
        separator="\t",
    )

    log("")
    log(f"TSV diagnostics written to:")
    log(f"  {diagnostics_tsv}")

    # --------------------------------------------------------
    # 12. Write summaries
    # --------------------------------------------------------

    write_summary(
        df,
        output_dir,
        args.cap,
    )

    # --------------------------------------------------------
    # 13. Print important examples
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("TOP POSSIBLE CAP-RELATED MISSES")
    log("=" * 70)

    examples = (
        df.filter(
            pl.col("strong_cap_evidence")
        )
        .select(
            [
                "s1_id",
                "match_id",
                "match_source",
                "s1_name",
                "match_name",
                "s1_name_core",
                "match_name_core",
                "name_total_pairs",
                "core_total_pairs",
                "name_exact_cap_risk",
                "name_core_cap_risk",
                "cap_analysis_category",
            ]
        )
        .sort(
            [
                "name_total_pairs",
                "core_total_pairs",
            ],
            descending=True,
        )
        .head(30)
    )

    if examples.height:
        print(examples)
    else:
        log("No strong cap-related examples found.")

    # --------------------------------------------------------
    # 14. Final message
    # --------------------------------------------------------

    log("")
    log("=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)

    log("")
    log("Created:")
    log(
        f"  {output_dir / 'block_cap_diagnostics.parquet'}"
    )
    log(
        f"  {output_dir / 'block_cap_diagnostics.tsv'}"
    )
    log(
        f"  {output_dir / 'block_cap_summary.tsv'}"
    )
    log(
        f"  {output_dir / 'block_cap_category_summary.tsv'}"
    )

    log("")
    log("IMPORTANT:")
    log(
        "Do NOT change MAX_PAIRS_PER_BLOCK_KEY yet."
    )
    log(
        "First inspect block_cap_summary.tsv and "
        "block_cap_category_summary.tsv."
    )
    log(
        "We will use those results to decide whether "
        "the blocking strategy should be changed."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\nInterrupted by user.")
        sys.exit(130)
    except Exception as exc:
        log("")
        log("=" * 70)
        log("ERROR")
        log("=" * 70)
        log(str(exc))
        sys.exit(1)
