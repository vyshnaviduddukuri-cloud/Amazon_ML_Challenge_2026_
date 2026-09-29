"""
Analyze missed true pairs from the blocking stage.

Purpose
-------
The current blocking stage found ~78.19% of true pairs.
This script analyzes the missed true pairs so we can design
additional targeted blocking passes.

Directory structure expected
----------------------------
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
│
├── normalized/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   └── train_source3.tsv
│
└── candidates/
    └── candidate_partitions/
        ├── candidate_part_000.parquet
        ├── ...
        └── candidate_part_063.parquet

Outputs
-------
dataset/candidates/missed_analysis/
├── missed_true_pairs.parquet
├── missed_sample.parquet
├── missed_pair_diagnostics.tsv
├── missed_summary.tsv
└── analysis_summary.tsv

Run
---
python3 analyze_missed_pairs.py

Optional
--------
python3 analyze_missed_pairs.py --sample-size 20000
"""

import argparse
import gc
import math
import re
import unicodedata
from pathlib import Path

import polars as pl


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_TRAIN_DIR = Path("dataset/train")
DEFAULT_NORMALIZED_DIR = Path("dataset/normalized")
DEFAULT_CANDIDATE_DIR = Path("dataset/candidates")
DEFAULT_OUTPUT_DIR = Path("dataset/candidates/missed_analysis")

DEFAULT_SAMPLE_SIZE = 20_000


# ============================================================================
# Validation helpers
# ============================================================================

def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {description}:\n"
            f"  {path}\n"
        )


def require_columns(df_or_lf, required_columns, description: str) -> None:
    columns = df_or_lf.collect_schema().names()

    missing = [c for c in required_columns if c not in columns]

    if missing:
        raise ValueError(
            f"\n{description} is missing required columns:\n"
            f"  {missing}\n"
            f"Available columns:\n"
            f"  {columns}\n"
        )


# ============================================================================
# Ground truth
# ============================================================================

def build_true_pairs(gt_path: Path) -> pl.LazyFrame:
    """
    Convert:

        source1_entity_id    matched_entity_ids

    into:

        source1_entity_id    candidate_entity_id

    without creating millions of Python dictionaries/lists.
    """

    gt_lf = pl.scan_csv(
        gt_path,
        separator="\t",
        infer_schema_length=10_000,
        null_values=[""],
    )

    require_columns(
        gt_lf,
        ["source1_entity_id", "matched_entity_ids"],
        "training ground truth",
    )

    true_pairs = (
        gt_lf
        .with_columns(
            pl.col("matched_entity_ids")
            .fill_null("")
            .str.strip_chars()
            .alias("_matched")
        )
        .filter(pl.col("_matched") != "")
        .with_columns(
            pl.col("_matched")
            .str.split(",")
            .alias("_ids")
        )
        .explode("_ids")
        .with_columns(
            pl.col("_ids")
            .str.strip_chars()
            .alias("candidate_entity_id")
        )
        .filter(pl.col("candidate_entity_id") != "")
        .select(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .unique()
    )

    return true_pairs


# ============================================================================
# Candidate partitions
# ============================================================================

def get_candidate_partitions(candidate_dir: Path) -> list[Path]:
    partition_dir = candidate_dir / "candidate_partitions"

    require_path(
        partition_dir,
        "candidate partition directory",
    )

    partitions = sorted(
        partition_dir.glob("candidate_part_*.parquet")
    )

    if not partitions:
        raise FileNotFoundError(
            f"\nNo candidate partition files found in:\n"
            f"  {partition_dir}\n"
        )

    return partitions


def validate_candidate_partition(path: Path) -> None:
    lf = pl.scan_parquet(path)

    require_columns(
        lf,
        [
            "source1_entity_id",
            "candidate_entity_id",
        ],
        f"candidate partition {path.name}",
    )


# ============================================================================
# Find missed true pairs
# ============================================================================

def find_missed_true_pairs(
    true_pairs_lf: pl.LazyFrame,
    candidate_partitions: list[Path],
    output_path: Path,
) -> tuple[int, int, int]:
    """
    Determine which ground-truth pairs are absent from ALL candidate
    partitions.

    Important:
    We process candidate partitions one at a time.

    We do NOT create a giant in-memory DataFrame containing all
    ~205 million candidate pairs.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n=== Finding missed true pairs ===")
    print(f"Candidate partitions: {len(candidate_partitions)}")

    # ------------------------------------------------------------------
    # First write the complete true-pair table to disk.
    # This gives us a reusable Parquet representation.
    # ------------------------------------------------------------------

    true_pairs_path = output_path.parent / "_true_pairs.parquet"

    print("\nPreparing ground-truth pair table...")

    (
        true_pairs_lf
        .sink_parquet(true_pairs_path)
    )

    true_pairs_disk = pl.scan_parquet(true_pairs_path)

    total_true = (
        true_pairs_disk
        .select(pl.len())
        .collect()
        .item()
    )

    print(f"True pairs: {total_true:,}")

    # ------------------------------------------------------------------
    # Instead of joining all 205M candidates at once, determine found
    # true pairs partition by partition.
    #
    # We accumulate only the TRUE PAIRS that were found, not all
    # candidate pairs.
    # ------------------------------------------------------------------

    found_dir = output_path.parent / "_found_true_pair_parts"

    if found_dir.exists():
        for old_file in found_dir.glob("*.parquet"):
            old_file.unlink()
    else:
        found_dir.mkdir(parents=True)

    total_found = 0

    for index, partition_path in enumerate(candidate_partitions, start=1):

        print(
            f"\n[{index}/{len(candidate_partitions)}] "
            f"Checking {partition_path.name}..."
        )

        validate_candidate_partition(partition_path)

        candidates = (
            pl.scan_parquet(partition_path)
            .select(
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ]
            )
            .unique()
        )

        # Only rows that are both:
        #   ground-truth pair
        #   candidate pair
        #
        # The result is at most the number of true pairs (~7.6M),
        # not the full candidate table.
        found = (
            true_pairs_disk
            .join(
                candidates,
                on=[
                    "source1_entity_id",
                    "candidate_entity_id",
                ],
                how="inner",
            )
            .unique()
        )

        found_path = found_dir / f"found_{index:03d}.parquet"

        found.sink_parquet(found_path)

        part_found = (
            pl.scan_parquet(found_path)
            .select(pl.len())
            .collect()
            .item()
        )

        total_found += part_found

        print(
            f"  true pairs found in this partition: "
            f"{part_found:,}"
        )

        del candidates
        del found

        gc.collect()

    # ------------------------------------------------------------------
    # Union only the small found-true-pair files.
    # ------------------------------------------------------------------

    print("\nCombining found true pairs...")

    found_files = sorted(found_dir.glob("found_*.parquet"))

    if found_files:
        found_all = (
            pl.concat(
                [pl.scan_parquet(p) for p in found_files],
                how="vertical",
            )
            .unique()
        )

        found_all.sink_parquet(
            output_path.parent / "_found_true_pairs.parquet"
        )

        found_disk = pl.scan_parquet(
            output_path.parent / "_found_true_pairs.parquet"
        )

        unique_found = (
            found_disk
            .select(pl.len())
            .collect()
            .item()
        )

    else:
        unique_found = 0

    # ------------------------------------------------------------------
    # Anti-join ground truth against found true pairs.
    # ------------------------------------------------------------------

    print("\nFinding missed true pairs...")

    found_path = output_path.parent / "_found_true_pairs.parquet"

    if found_path.exists():
        found_disk = pl.scan_parquet(found_path)

        missed = (
            true_pairs_disk
            .join(
                found_disk,
                on=[
                    "source1_entity_id",
                    "candidate_entity_id",
                ],
                how="anti",
            )
        )

    else:
        missed = true_pairs_disk

    missed.sink_parquet(output_path)

    missed_count = (
        pl.scan_parquet(output_path)
        .select(pl.len())
        .collect()
        .item()
    )

    # ------------------------------------------------------------------
    # Cleanup intermediate files.
    # ------------------------------------------------------------------

    for p in found_dir.glob("*.parquet"):
        p.unlink()

    try:
        found_dir.rmdir()
    except OSError:
        pass

    if true_pairs_path.exists():
        true_pairs_path.unlink()

    if found_path.exists():
        found_path.unlink()

    print("\n=== Blocking recall result ===")
    print(f"True pairs : {total_true:,}")
    print(f"Found      : {unique_found:,}")
    print(f"Missed     : {missed_count:,}")

    if total_true:
        recall = unique_found / total_true
        print(f"Recall     : {recall:.6%}")

    return total_true, unique_found, missed_count


# ============================================================================
# Text normalization used ONLY for diagnostics
# ============================================================================

def diagnostic_clean(value: str) -> str:
    """
    Lightweight diagnostic normalization.

    This is NOT replacing the official normalization pipeline.
    It is only used to understand why pairs were missed.
    """

    if value is None:
        return ""

    text = str(value)

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        ch for ch in text
        if not unicodedata.combining(ch)
    )

    text = text.casefold()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


# ============================================================================
# Similarity helpers
# ============================================================================

def token_set(text: str) -> set[str]:
    if not text:
        return set()

    return {
        token
        for token in text.split()
        if token
    }


def jaccard_tokens(a: str, b: str) -> float:
    a_tokens = token_set(a)
    b_tokens = token_set(b)

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)

    if union == 0:
        return 0.0

    return intersection / union


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = text.replace(" ", "")

    if not text:
        return set()

    if len(text) < n:
        return {text}

    return {
        text[i:i + n]
        for i in range(len(text) - n + 1)
    }


def jaccard_ngrams(a: str, b: str, n: int = 3) -> float:
    a_set = char_ngrams(a, n)
    b_set = char_ngrams(b, n)

    if not a_set and not b_set:
        return 1.0

    if not a_set or not b_set:
        return 0.0

    union = a_set | b_set

    if not union:
        return 0.0

    return len(a_set & b_set) / len(union)


# ============================================================================
# Load source records
# ============================================================================

def load_normalized_source(
    normalized_dir: Path,
    filename: str,
    source_label: str,
) -> pl.LazyFrame:

    path = normalized_dir / filename

    require_path(
        path,
        f"normalized {source_label} file",
    )

    lf = pl.scan_csv(
        path,
        separator="\t",
        infer_schema_length=10_000,
        null_values=[""],
    )

    required = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
        "normalized_name",
        "normalized_name_core",
        "normalized_address",
        "normalized_country",
    ]

    require_columns(
        lf,
        required,
        f"normalized {source_label} file",
    )

    return lf


# ============================================================================
# Sample missed pairs
# ============================================================================

def create_missed_sample(
    missed_path: Path,
    output_dir: Path,
    sample_size: int,
) -> Path:

    output_path = output_dir / "missed_sample.parquet"

    print(
        f"\nCreating diagnostic sample "
        f"({sample_size:,} missed pairs)..."
    )

    missed = pl.scan_parquet(missed_path)

    # Deterministic sample:
    # sorting by IDs and taking the first N avoids random-memory
    # behavior and makes reruns reproducible.
    sample = (
        missed
        .sort(
            [
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )
        .head(sample_size)
    )

    sample.sink_parquet(output_path)

    actual = (
        pl.scan_parquet(output_path)
        .select(pl.len())
        .collect()
        .item()
    )

    print(f"Sample written: {actual:,} rows")

    return output_path


# ============================================================================
# Join missed sample to source records
# ============================================================================

def build_diagnostic_pairs(
    sample_path: Path,
    normalized_dir: Path,
    output_dir: Path,
) -> Path:

    print("\nLoading normalized source data for diagnostics...")

    s1 = load_normalized_source(
        normalized_dir,
        "train_source1.tsv",
        "S1",
    )

    s2 = load_normalized_source(
        normalized_dir,
        "train_source2.tsv",
        "S2",
    )

    s3 = load_normalized_source(
        normalized_dir,
        "train_source3.tsv",
        "S3",
    )

    sample = pl.scan_parquet(sample_path)

    # ---------------------------------------------------------------
    # S1 record
    # ---------------------------------------------------------------

    s1_small = (
        s1
        .select(
            [
                pl.col("entity_id").alias("source1_entity_id"),

                pl.col("business_name")
                .alias("business_name_s1"),

                pl.col("business_address")
                .alias("business_address_s1"),

                pl.col("country")
                .alias("country_s1"),

                pl.col("normalized_name")
                .alias("normalized_name_s1"),

                pl.col("normalized_name_core")
                .alias("normalized_name_core_s1"),

                pl.col("normalized_address")
                .alias("normalized_address_s1"),

                pl.col("normalized_country")
                .alias("normalized_country_s1"),
            ]
        )
    )

    # ---------------------------------------------------------------
    # S2 record
    # ---------------------------------------------------------------

    s2_small = (
        s2
        .select(
            [
                pl.col("entity_id")
                .alias("candidate_entity_id"),

                pl.col("business_name")
                .alias("business_name_other"),

                pl.col("business_address")
                .alias("business_address_other"),

                pl.col("country")
                .alias("country_other"),

                pl.col("normalized_name")
                .alias("normalized_name_other"),

                pl.col("normalized_name_core")
                .alias("normalized_name_core_other"),

                pl.col("normalized_address")
                .alias("normalized_address_other"),

                pl.col("normalized_country")
                .alias("normalized_country_other"),
            ]
        )
        .with_columns(
            pl.lit("S2").alias("candidate_source")
        )
    )

    # ---------------------------------------------------------------
    # S3 record
    # ---------------------------------------------------------------

    s3_small = (
        s3
        .select(
            [
                pl.col("entity_id")
                .alias("candidate_entity_id"),

                pl.col("business_name")
                .alias("business_name_other"),

                pl.col("business_address")
                .alias("business_address_other"),

                pl.col("country")
                .alias("country_other"),

                pl.col("normalized_name")
                .alias("normalized_name_other"),

                pl.col("normalized_name_core")
                .alias("normalized_name_core_other"),

                pl.col("normalized_address")
                .alias("normalized_address_other"),

                pl.col("normalized_country")
                .alias("normalized_country_other"),
            ]
        )
        .with_columns(
            pl.lit("S3").alias("candidate_source")
        )
    )

    other = pl.concat(
        [s2_small, s3_small],
        how="vertical",
    )

    diagnostics = (
        sample
        .join(
            s1_small,
            on="source1_entity_id",
            how="left",
        )
        .join(
            other,
            on="candidate_entity_id",
            how="left",
        )
    )

    output_path = output_dir / "missed_pair_diagnostics.parquet"

    diagnostics.sink_parquet(output_path)

    print(
        f"Diagnostic pair table written:\n"
        f"  {output_path}"
    )

    return output_path


# ============================================================================
# Compute diagnostics
# ============================================================================

def compute_pair_diagnostics(
    diagnostics_path: Path,
    output_dir: Path,
) -> None:

    print("\nComputing missed-pair diagnostics...")

    df = pl.read_parquet(diagnostics_path)

    if df.height == 0:
        print("No missed pairs to analyze.")
        return

    # ------------------------------------------------------------------
    # Basic diagnostic normalization.
    # ------------------------------------------------------------------

    df = df.with_columns(
        [
            pl.col("normalized_name_s1")
            .fill_null("")
            .str.strip_chars()
            .alias("_name1"),

            pl.col("normalized_name_other")
            .fill_null("")
            .str.strip_chars()
            .alias("_name2"),

            pl.col("normalized_name_core_s1")
            .fill_null("")
            .str.strip_chars()
            .alias("_core1"),

            pl.col("normalized_name_core_other")
            .fill_null("")
            .str.strip_chars()
            .alias("_core2"),

            pl.col("normalized_address_s1")
            .fill_null("")
            .str.strip_chars()
            .alias("_addr1"),

            pl.col("normalized_address_other")
            .fill_null("")
            .str.strip_chars()
            .alias("_addr2"),

            pl.col("normalized_country_s1")
            .fill_null("")
            .str.strip_chars()
            .alias("_country1"),

            pl.col("normalized_country_other")
            .fill_null("")
            .str.strip_chars()
            .alias("_country2"),
        ]
    )

    # ------------------------------------------------------------------
    # Exact equality diagnostics.
    # ------------------------------------------------------------------

    df = df.with_columns(
        [
            (
                (pl.col("_name1") != "")
                & (pl.col("_name2") != "")
                & (pl.col("_name1") == pl.col("_name2"))
            ).alias("name_exact"),

            (
                (pl.col("_core1") != "")
                & (pl.col("_core2") != "")
                & (pl.col("_core1") == pl.col("_core2"))
            ).alias("name_core_exact"),

            (
                (pl.col("_addr1") != "")
                & (pl.col("_addr2") != "")
                & (pl.col("_addr1") == pl.col("_addr2"))
            ).alias("address_exact"),

            (
                (pl.col("_country1") != "")
                & (pl.col("_country2") != "")
                & (pl.col("_country1") == pl.col("_country2"))
            ).alias("country_exact"),

            (
                (pl.col("_addr1") == "")
                | (pl.col("_addr2") == "")
            ).alias("address_missing"),

            (
                pl.col("_addr1") == ""
            ).alias("address_missing_s1"),

            (
                pl.col("_addr2") == ""
            ).alias("address_missing_other"),
        ]
    )

    # ------------------------------------------------------------------
    # Python-level similarity calculations.
    #
    # We only do this on the diagnostic sample, not the full
    # 1.66M missed pairs.
    # ------------------------------------------------------------------

    name_token_scores = []
    address_token_scores = []
    name_tri_scores = []
    address_tri_scores = []

    name_lengths_1 = []
    name_lengths_2 = []
    address_lengths_1 = []
    address_lengths_2 = []

    for row in df.select(
        [
            "_name1",
            "_name2",
            "_addr1",
            "_addr2",
        ]
    ).iter_rows(named=True):

        name1 = row["_name1"] or ""
        name2 = row["_name2"] or ""

        addr1 = row["_addr1"] or ""
        addr2 = row["_addr2"] or ""

        name_token_scores.append(
            jaccard_tokens(name1, name2)
        )

        address_token_scores.append(
            jaccard_tokens(addr1, addr2)
        )

        name_tri_scores.append(
            jaccard_ngrams(name1, name2, 3)
        )

        address_tri_scores.append(
            jaccard_ngrams(addr1, addr2, 3)
        )

        name_lengths_1.append(len(name1))
        name_lengths_2.append(len(name2))

        address_lengths_1.append(len(addr1))
        address_lengths_2.append(len(addr2))

    df = df.with_columns(
        [
            pl.Series(
                "name_token_jaccard",
                name_token_scores,
            ),

            pl.Series(
                "address_token_jaccard",
                address_token_scores,
            ),

            pl.Series(
                "name_trigram_jaccard",
                name_tri_scores,
            ),

            pl.Series(
                "address_trigram_jaccard",
                address_tri_scores,
            ),

            pl.Series(
                "name_length_s1",
                name_lengths_1,
            ),

            pl.Series(
                "name_length_other",
                name_lengths_2,
            ),

            pl.Series(
                "address_length_s1",
                address_lengths_1,
            ),

            pl.Series(
                "address_length_other",
                address_lengths_2,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Useful blocking-style signals.
    # ------------------------------------------------------------------

    df = df.with_columns(
        [
            (
                pl.col("name_token_jaccard") >= 0.50
            ).alias("name_token_jaccard_ge_050"),

            (
                pl.col("name_token_jaccard") >= 0.70
            ).alias("name_token_jaccard_ge_070"),

            (
                pl.col("name_token_jaccard") >= 0.80
            ).alias("name_token_jaccard_ge_080"),

            (
                pl.col("name_trigram_jaccard") >= 0.50
            ).alias("name_trigram_jaccard_ge_050"),

            (
                pl.col("name_trigram_jaccard") >= 0.70
            ).alias("name_trigram_jaccard_ge_070"),

            (
                pl.col("address_token_jaccard") >= 0.50
            ).alias("address_token_jaccard_ge_050"),

            (
                pl.col("address_trigram_jaccard") >= 0.50
            ).alias("address_trigram_jaccard_ge_050"),
        ]
    )

    # ------------------------------------------------------------------
    # Categorize why a pair may have escaped current blocking.
    # ------------------------------------------------------------------

    categories = []

    for row in df.select(
        [
            "name_exact",
            "name_core_exact",
            "address_exact",
            "country_exact",
            "address_missing",
            "name_token_jaccard",
            "address_token_jaccard",
            "name_trigram_jaccard",
            "address_trigram_jaccard",
        ]
    ).iter_rows(named=True):

        name_exact = row["name_exact"]
        core_exact = row["name_core_exact"]
        addr_exact = row["address_exact"]

        name_tok = row["name_token_jaccard"]
        addr_tok = row["address_token_jaccard"]

        name_tri = row["name_trigram_jaccard"]
        addr_tri = row["address_trigram_jaccard"]

        if name_exact:
            category = "name_exact_but_missed"

        elif core_exact:
            category = "name_core_exact_but_missed"

        elif addr_exact:
            category = "address_exact_but_missed"

        elif name_tok >= 0.70 and name_tri >= 0.70:
            category = "strong_name_similarity"

        elif name_tok >= 0.50 or name_tri >= 0.50:
            category = "moderate_name_similarity"

        elif addr_tok >= 0.70 and addr_tri >= 0.70:
            category = "strong_address_similarity"

        elif addr_tok >= 0.50 or addr_tri >= 0.50:
            category = "moderate_address_similarity"

        elif row["address_missing"]:
            category = "missing_address"

        else:
            category = "weak_surface_similarity"

        categories.append(category)

    df = df.with_columns(
        pl.Series(
            "miss_category",
            categories,
        )
    )

    # ------------------------------------------------------------------
    # Write detailed diagnostics.
    # ------------------------------------------------------------------

    diagnostics_tsv = (
        output_dir / "missed_pair_diagnostics.tsv"
    )

    df.write_csv(
        diagnostics_tsv,
        separator="\t",
    )

    print(
        f"\nDetailed diagnostics written:\n"
        f"  {diagnostics_tsv}"
    )

    # ------------------------------------------------------------------
    # Summary table.
    # ------------------------------------------------------------------

    total = df.height

    summary_rows = []

    def add_summary(
        metric: str,
        count: int,
    ) -> None:
        pct = (
            count / total
            if total
            else 0.0
        )

        summary_rows.append(
            {
                "metric": metric,
                "count": count,
                "percentage": pct,
            }
        )

    add_summary(
        "name_exact",
        int(df["name_exact"].sum()),
    )

    add_summary(
        "name_core_exact",
        int(df["name_core_exact"].sum()),
    )

    add_summary(
        "address_exact",
        int(df["address_exact"].sum()),
    )

    add_summary(
        "country_exact",
        int(df["country_exact"].sum()),
    )

    add_summary(
        "address_missing",
        int(df["address_missing"].sum()),
    )

    add_summary(
        "name_token_jaccard_ge_050",
        int(df["name_token_jaccard_ge_050"].sum()),
    )

    add_summary(
        "name_token_jaccard_ge_070",
        int(df["name_token_jaccard_ge_070"].sum()),
    )

    add_summary(
        "name_token_jaccard_ge_080",
        int(df["name_token_jaccard_ge_080"].sum()),
    )

    add_summary(
        "name_trigram_jaccard_ge_050",
        int(df["name_trigram_jaccard_ge_050"].sum()),
    )

    add_summary(
        "name_trigram_jaccard_ge_070",
        int(df["name_trigram_jaccard_ge_070"].sum()),
    )

    add_summary(
        "address_token_jaccard_ge_050",
        int(df["address_token_jaccard_ge_050"].sum()),
    )

    add_summary(
        "address_trigram_jaccard_ge_050",
        int(df["address_trigram_jaccard_ge_050"].sum()),
    )

    summary = pl.DataFrame(summary_rows)

    summary_path = (
        output_dir / "missed_summary.tsv"
    )

    summary.write_csv(
        summary_path,
        separator="\t",
    )

    # ------------------------------------------------------------------
    # Category distribution.
    # ------------------------------------------------------------------

    category_summary = (
        df
        .group_by("miss_category")
        .agg(
            pl.len().alias("count")
        )
        .with_columns(
            (
                pl.col("count") / total
            ).alias("percentage")
        )
        .sort(
            "count",
            descending=True,
        )
    )

    category_path = (
        output_dir / "analysis_summary.tsv"
    )

    category_summary.write_csv(
        category_path,
        separator="\t",
    )

    # ------------------------------------------------------------------
    # Country/source breakdown.
    # ------------------------------------------------------------------

    source_summary = (
        df
        .group_by(
            [
                "candidate_source",
                "normalized_country_s1",
                "normalized_country_other",
            ]
        )
        .agg(
            pl.len().alias("count")
        )
        .sort(
            "count",
            descending=True,
        )
    )

    source_path = (
        output_dir / "missed_source_country_summary.tsv"
    )

    source_summary.write_csv(
        source_path,
        separator="\t",
    )

    # ------------------------------------------------------------------
    # Print concise report.
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSED-PAIR DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(f"\nDiagnostic sample: {total:,}")

    print("\nExact matching signals:")

    for metric in [
        "name_exact",
        "name_core_exact",
        "address_exact",
        "country_exact",
    ]:
        row = summary.filter(
            pl.col("metric") == metric
        )

        if row.height:
            count = row["count"][0]
            pct = row["percentage"][0]

            print(
                f"  {metric:30s}: "
                f"{count:>8,} ({pct:.2%})"
            )

    print("\nName similarity:")

    for metric in [
        "name_token_jaccard_ge_050",
        "name_token_jaccard_ge_070",
        "name_token_jaccard_ge_080",
        "name_trigram_jaccard_ge_050",
        "name_trigram_jaccard_ge_070",
    ]:
        row = summary.filter(
            pl.col("metric") == metric
        )

        if row.height:
            count = row["count"][0]
            pct = row["percentage"][0]

            print(
                f"  {metric:30s}: "
                f"{count:>8,} ({pct:.2%})"
            )

    print("\nAddress similarity:")

    for metric in [
        "address_token_jaccard_ge_050",
        "address_trigram_jaccard_ge_050",
    ]:
        row = summary.filter(
            pl.col("metric") == metric
        )

        if row.height:
            count = row["count"][0]
            pct = row["percentage"][0]

            print(
                f"  {metric:30s}: "
                f"{count:>8,} ({pct:.2%})"
            )

    print("\nMiss categories:")

    print(category_summary)

    print("\nOutput files:")
    print(f"  {summary_path}")
    print(f"  {category_path}")
    print(f"  {source_path}")
    print(f"  {diagnostics_tsv}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Analyze true pairs missed by the current blocking stage."
        )
    )

    parser.add_argument(
        "--train-dir",
        default=str(DEFAULT_TRAIN_DIR),
        help=(
            "Directory containing train_ground_truth.tsv "
            "(default: dataset/train)"
        ),
    )

    parser.add_argument(
        "--normalized-dir",
        default=str(DEFAULT_NORMALIZED_DIR),
        help=(
            "Directory containing normalized train source TSVs "
            "(default: dataset/normalized)"
        ),
    )

    parser.add_argument(
        "--candidate-dir",
        default=str(DEFAULT_CANDIDATE_DIR),
        help=(
            "Directory containing candidate_partitions "
            "(default: dataset/candidates)"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Directory for analysis outputs "
            "(default: dataset/candidates/missed_analysis)"
        ),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=(
            "Number of missed pairs to use for detailed diagnostics "
            "(default: 20000)"
        ),
    )

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    normalized_dir = Path(args.normalized_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)

    if args.sample_size <= 0:
        raise ValueError(
            "--sample-size must be greater than 0"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    gt_path = (
        train_dir / "train_ground_truth.tsv"
    )

    require_path(
        gt_path,
        "training ground truth",
    )

    require_path(
        normalized_dir / "train_source1.tsv",
        "normalized train source1",
    )

    require_path(
        normalized_dir / "train_source2.tsv",
        "normalized train source2",
    )

    require_path(
        normalized_dir / "train_source3.tsv",
        "normalized train source3",
    )

    candidate_partitions = (
        get_candidate_partitions(candidate_dir)
    )

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------

    print("=" * 70)
    print("MISSED TRUE-PAIR ANALYSIS")
    print("=" * 70)

    print(f"\nGround truth:")
    print(f"  {gt_path}")

    print(f"\nNormalized sources:")
    print(f"  {normalized_dir}")

    print(f"\nCandidate partitions:")
    print(f"  {candidate_dir / 'candidate_partitions'}")
    print(f"  {len(candidate_partitions)} partitions")

    print(f"\nOutput:")
    print(f"  {output_dir}")

    print(f"\nDiagnostic sample size:")
    print(f"  {args.sample_size:,}")

    # ------------------------------------------------------------------
    # Build ground truth
    # ------------------------------------------------------------------

    true_pairs_lf = build_true_pairs(
        gt_path
    )

    # ------------------------------------------------------------------
    # Find missed true pairs
    # ------------------------------------------------------------------

    missed_path = (
        output_dir / "missed_true_pairs.parquet"
    )

    total_true, found, missed = (
        find_missed_true_pairs(
            true_pairs_lf,
            candidate_partitions,
            missed_path,
        )
    )

    # ------------------------------------------------------------------
    # Basic recall
    # ------------------------------------------------------------------

    recall = (
        found / total_true
        if total_true
        else float("nan")
    )

    print("\n" + "=" * 70)
    print("FINAL BLOCKING STATISTICS")
    print("=" * 70)

    print(
        f"True pairs       : {total_true:,}"
    )

    print(
        f"Found by blocking: {found:,}"
    )

    print(
        f"Missed           : {missed:,}"
    )

    print(
        f"Blocking recall  : {recall:.6%}"
    )

    print(
        f"\nMissed pairs saved to:"
        f"\n  {missed_path}"
    )

    # ------------------------------------------------------------------
    # If there are no misses, stop.
    # ------------------------------------------------------------------

    if missed == 0:
        print(
            "\nNo missed true pairs. "
            "Nothing further to analyze."
        )
        return

    # ------------------------------------------------------------------
    # Create diagnostic sample
    # ------------------------------------------------------------------

    sample_path = create_missed_sample(
        missed_path,
        output_dir,
        args.sample_size,
    )

    # ------------------------------------------------------------------
    # Join sample against normalized source records
    # ------------------------------------------------------------------

    diagnostics_path = build_diagnostic_pairs(
        sample_path,
        normalized_dir,
        output_dir,
    )

    # ------------------------------------------------------------------
    # Calculate diagnostics
    # ------------------------------------------------------------------

    compute_pair_diagnostics(
        diagnostics_path,
        output_dir,
    )

    # ------------------------------------------------------------------
    # Final message
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\nNext step:"
    )

    print(
        "Read these two files first:"
    )

    print(
        f"  cat {output_dir / 'missed_summary.tsv'}"
    )

    print(
        f"  cat {output_dir / 'analysis_summary.tsv'}"
    )

    print(
        "\nWe will use those results to design targeted "
        "additional blocking passes."
    )


if __name__ == "__main__":
    main()
