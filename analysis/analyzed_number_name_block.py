import os
import re
import polars as pl

# ============================================================
# NUMBER + NAME-TOKEN SHADOW BLOCKING EXPERIMENT
# ============================================================
#
# Shadow experiment ONLY.
#
# Does NOT modify build_candidates_5.py.
#
# Blocking key:
#
#     country + informative_name_token + address_number
#
# The experiment measures:
#
#   1. Total missed true pairs
#   2. Missed pairs recovered
#   3. Recovery percentage
#   4. Estimated candidate pairs
#   5. Candidates per recovered pair
#   6. Block-size statistics
#
# IMPORTANT:
# The experiment uses the EXISTING candidate partitions and
# ground truth to identify the currently missed true pairs.
# ============================================================


# ============================================================
# PATHS
# ============================================================

NORMALIZED_DIR = "dataset/normalized"

GT_FILE = "dataset/train/train_ground_truth.tsv"

CANDIDATE_DIR = "dataset/candidates"

OUTPUT_DIR = os.path.join(
    CANDIDATE_DIR,
    "missed_analysis"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "number_name_block_analysis.tsv"
)

NUM_PARTITIONS = 64

PAIR_CAP = 5000

NAME_STOPWORDS = {
    "the",
    "and",
    "of",
    "for",
    "new",
    "inc",
    "llc",
    "ltd",
    "co",
    "company",
    "corporation",
    "corp",
    "group",
    "international",
    "intl",
    "private",
    "limited",
    "pvt",
    "enterprises",
    "enterprise",
    "sarl",
    "sas",
    "sa",
}


# ============================================================
# HELPERS
# ============================================================

def candidate_partition_path(partition):
    return os.path.join(
        CANDIDATE_DIR,
        "candidate_partitions",
        f"candidate_part_{partition:03d}.parquet"
    )


def find_candidate_file(partition):
    """
    Support the existing candidate-partition naming convention.
    """

    candidates = [
        candidate_partition_path(partition),
        os.path.join(
            CANDIDATE_DIR,
            f"candidates_part_{partition:03d}.parquet"
        ),
        os.path.join(
            CANDIDATE_DIR,
            f"candidate_pairs_{partition:03d}.parquet"
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def scan_source(filename):
    return pl.scan_csv(
        os.path.join(
            NORMALIZED_DIR,
            filename
        ),
        separator="\t",
        infer_schema_length=1000,
        null_values=[""],
    )


def extract_address_number_expr(column):
    """
    Extract the first numeric sequence from an address.

    Examples:
        "1016 main street" -> "1016"
        "12-14 market road" -> "12"
        "no number here" -> ""
    """

    return (
        pl.col(column)
        .fill_null("")
        .str.extract(
            r"(?:^|[^0-9])([0-9]{1,6})(?:[^0-9]|$)",
            group_index=1,
        )
        .fill_null("")
    )

def make_number_name_keys(lf):
    """
    Create one row per entity + informative name token +
    address number.
    """

    return (
        lf
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "normalized_address",
        ])
        .filter(
            pl.col("normalized_name_core")
            .fill_null("")
            != ""
        )
        .with_columns([
            pl.col("normalized_name_core")
            .fill_null("")
            .str.split(" ")
            .alias("name_tokens"),

            extract_address_number_expr(
                "normalized_address"
            ).alias("address_number"),
        ])
        .explode("name_tokens")
        .rename({
            "name_tokens": "name_token"
        })
        .filter(
            pl.col("name_token") != ""
        )
        .filter(
            ~pl.col("name_token").is_in(
                NAME_STOPWORDS
            )
        )
        .filter(
            pl.col("name_token").str.len_chars() >= 3
        )
        .filter(
            pl.col("address_number") != ""
        )
        .select([
            "entity_id",
            "normalized_country",
            "name_token",
            "address_number",
        ])
        .unique()
    )


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NUMBER + NAME-TOKEN SHADOW BLOCKING EXPERIMENT")
print("=" * 70)


# ============================================================
# CHECK CANDIDATE PARTITIONS
# ============================================================

print("\nChecking candidate partitions...")

candidate_files = []

for partition in range(NUM_PARTITIONS):

    path = find_candidate_file(partition)

    if path is None:
        raise FileNotFoundError(
            f"Could not find candidate partition "
            f"{partition:03d}."
        )

    candidate_files.append(path)

print(
    f"Found {len(candidate_files):,}/"
    f"{NUM_PARTITIONS} candidate partitions."
)


# ============================================================
# LOAD TRAIN SOURCES
# ============================================================

print("\nLoading normalized sources...")

s1 = scan_source(
    "train_source1.tsv"
)

s2 = scan_source(
    "train_source2.tsv"
)

s3 = scan_source(
    "train_source3.tsv"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pl.scan_csv(
    GT_FILE,
    separator="\t",
    infer_schema_length=1000,
    null_values=[""],
)

gt_pairs = (
    gt
    .select([
        "source1_entity_id",
        "matched_entity_ids",
    ])
    .filter(
        pl.col("matched_entity_ids")
        .fill_null("")
        != ""
    )
    .with_columns(
        pl.col("matched_entity_ids")
        .str.split(",")
        .alias("match_ids")
    )
    .explode("match_ids")
    .rename({
        "source1_entity_id": "s1_id",
        "match_ids": "match_id",
    })
    .select([
        "s1_id",
        "match_id",
    ])
    .unique()
)


# ============================================================
# FIND CURRENTLY MISSED TRUE PAIRS
# ============================================================
#
# Process one candidate partition at a time.
#
# This avoids loading the ~205M candidate pairs into memory.
# ============================================================

print("\nFinding currently missed true pairs...")

missed_parts = []

total_gt_pairs = 0
total_found_pairs = 0

for partition, candidate_file in enumerate(
    candidate_files
):

    print(
        f"\rProcessing partition "
        f"{partition + 1:02d}/{NUM_PARTITIONS}...",
        end="",
        flush=True,
    )

    gt_part = (
    gt_pairs
    .filter(
        pl.col("s1_id")
        .hash()
        .mod(NUM_PARTITIONS)
        == partition
    )
)

    candidate_part = pl.scan_parquet(
        candidate_file
    )

    # --------------------------------------------------------
    # Detect actual column names.
    # --------------------------------------------------------

    schema = candidate_part.collect_schema()

    columns = schema.names()

    if "s1_id" in columns:
        candidate_s1 = "s1_id"
    elif "source1_entity_id" in columns:
        candidate_s1 = "source1_entity_id"
    else:
        raise ValueError(
            f"Could not identify S1 ID column in "
            f"{candidate_file}. Columns: {columns}"
        )

    if "candidate_entity_id" in columns:
        candidate_match = "candidate_entity_id"
    elif "candidate_id" in columns:
        candidate_match = "candidate_id"
    elif "match_id" in columns:
        candidate_match = "match_id"
    elif "entity_id" in columns:
        candidate_match = "entity_id"
    else:
        raise ValueError(
            f"Could not identify candidate ID column "
            f"in {candidate_file}. Columns: {columns}"
        )

    candidate_pairs = (
        candidate_part
        .select([
            pl.col(candidate_s1)
            .alias("s1_id"),

            pl.col(candidate_match)
            .alias("match_id"),
        ])
        .unique()
    )

    gt_count = gt_part.select(
        pl.len()
    ).collect().item()

    total_gt_pairs += gt_count

    found = (
        gt_part
        .join(
            candidate_pairs,
            on=[
                "s1_id",
                "match_id",
            ],
            how="semi",
        )
        .collect()
    )

    total_found_pairs += found.height

    missed = (
        gt_part
        .join(
            candidate_pairs,
            on=[
                "s1_id",
                "match_id",
            ],
            how="anti",
        )
        .collect()
    )

    if missed.height:
        missed_parts.append(missed)


print()

if not missed_parts:
    raise RuntimeError(
        "No missed true pairs were found. "
        "The current candidate set appears to contain "
        "all ground-truth pairs."
    )

missed_pairs = pl.concat(
    missed_parts
).unique()

print(
    f"Total true matched pairs: "
    f"{total_gt_pairs:,}"
)

print(
    f"Currently found by candidates: "
    f"{total_found_pairs:,}"
)

print(
    f"Currently missed true pairs: "
    f"{missed_pairs.height:,}"
)


# ============================================================
# AFFECTED S1 ENTITIES
# ============================================================

affected_s1 = (
    missed_pairs
    .select("s1_id")
    .unique()
)

print(
    f"Affected S1 entities: "
    f"{affected_s1.height:,}"
)


# ============================================================
# CREATE NUMBER + NAME KEYS
# ============================================================

print("\nCreating number + name-token keys...")

s1_keys = (
    s1
    .join(
        affected_s1.lazy(),
        left_on="entity_id",
        right_on="s1_id",
        how="inner",
    )
    .pipe(make_number_name_keys)
)

s2_keys = make_number_name_keys(s2)

s3_keys = make_number_name_keys(s3)


# ============================================================
# CALCULATE BLOCK STATISTICS
# ============================================================

def calculate_block_cost(
    s1_key_lf,
    other_key_lf,
    source_name,
):

    print(
        f"\nCalculating candidate cost for "
        f"{source_name}..."
    )

    key_cols = [
        "normalized_country",
        "name_token",
        "address_number",
    ]

    s1_counts = (
        s1_key_lf
        .group_by(key_cols)
        .agg(
            pl.len().alias("s1_count")
        )
    )

    other_counts = (
        other_key_lf
        .group_by(key_cols)
        .agg(
            pl.len().alias("other_count")
        )
    )

    stats = (
        s1_counts
        .join(
            other_counts,
            on=key_cols,
            how="inner",
        )
        .with_columns(
            (
                pl.col("s1_count")
                *
                pl.col("other_count")
            ).alias("pair_count")
        )
        .select([
            pl.len().alias("blocks"),

            pl.col("pair_count")
            .sum()
            .alias("candidate_pairs"),

            pl.col("pair_count")
            .max()
            .alias("largest_block"),

            (
                pl.col("pair_count")
                > PAIR_CAP
            )
            .sum()
            .alias("overflow_blocks"),
        ])
        .collect()
        .row(0, named=True)
    )

    print(
        f"Blocks:              "
        f"{stats['blocks']:,}"
    )

    print(
        f"Estimated candidates: "
        f"{stats['candidate_pairs']:,}"
    )

    print(
        f"Largest block:       "
        f"{stats['largest_block']:,}"
    )

    print(
        f"Overflow blocks:     "
        f"{stats['overflow_blocks']:,}"
    )

    return stats


s2_stats = calculate_block_cost(
    s1_keys,
    s2_keys,
    "S2",
)

s3_stats = calculate_block_cost(
    s1_keys,
    s3_keys,
    "S3",
)


# ============================================================
# RECOVERY TEST
# ============================================================

print("\n" + "=" * 70)
print("TESTING RECOVERY")
print("=" * 70)


def test_recovery(
    missed,
    s1_key_lf,
    other_key_lf,
    other_source_name,
):

    source_prefix = (
        "S2"
        if other_source_name == "S2"
        else "S3"
    )

    source_missed = (
        missed
        .filter(
            pl.col("match_id")
            .str.starts_with(
                source_prefix + "-"
            )
        )
    )

    if source_missed.height == 0:
        return 0

    print(
        f"\nTesting {other_source_name}: "
        f"{source_missed.height:,} missed pairs"
    )

    s1_ids = (
        source_missed
        .select("s1_id")
        .unique()
        .get_column("s1_id")
        .to_list()
    )

    match_ids = (
        source_missed
        .select("match_id")
        .unique()
        .get_column("match_id")
        .to_list()
    )

    true_s1_keys = (
        s1_key_lf
        .filter(
            pl.col("entity_id")
            .is_in(s1_ids)
        )
        .select([
            pl.col("entity_id")
            .alias("s1_id"),
            "normalized_country",
            "name_token",
            "address_number",
        ])
        .collect()
    )

    true_other_keys = (
        other_key_lf
        .filter(
            pl.col("entity_id")
            .is_in(match_ids)
        )
        .select([
            pl.col("entity_id")
            .alias("match_id"),
            "normalized_country",
            "name_token",
            "address_number",
        ])
        .collect()
    )

    recovered = (
        source_missed
        .join(
            true_s1_keys,
            on="s1_id",
            how="inner",
        )
        .join(
            true_other_keys,
            on=[
                "match_id",
                "normalized_country",
                "name_token",
                "address_number",
            ],
            how="inner",
        )
        .select([
            "s1_id",
            "match_id",
        ])
        .unique()
    )

    count = recovered.height

    print(
        f"Recovered by number + name-token: "
        f"{count:,}"
    )

    return count


s2_recovered = test_recovery(
    missed_pairs,
    s1_keys,
    s2_keys,
    "S2",
)

s3_recovered = test_recovery(
    missed_pairs,
    s1_keys,
    s3_keys,
    "S3",
)


# ============================================================
# FINAL RESULTS
# ============================================================

total_recovered = (
    s2_recovered
    + s3_recovered
)

total_candidates = (
    s2_stats["candidate_pairs"]
    + s3_stats["candidate_pairs"]
)

recovery_percent = (
    100.0
    * total_recovered
    / missed_pairs.height
)

candidates_per_recovered = (
    total_candidates / total_recovered
    if total_recovered
    else None
)


print("\n" + "=" * 70)
print("NUMBER + NAME-TOKEN BLOCK SUMMARY")
print("=" * 70)

print(
    f"\nCurrently missed true pairs: "
    f"{missed_pairs.height:,}"
)

print(
    f"Recovered: "
    f"{total_recovered:,}"
)

print(
    f"Recovery percentage: "
    f"{recovery_percent:.4f}%"
)

print(
    f"\nEstimated candidate pairs: "
    f"{total_candidates:,}"
)

if candidates_per_recovered is not None:
    print(
        f"Candidates per recovered pair: "
        f"{candidates_per_recovered:,.2f}"
    )
else:
    print(
        "Candidates per recovered pair: N/A"
    )

print(
    f"\nS2 candidates: "
    f"{s2_stats['candidate_pairs']:,}"
)

print(
    f"S3 candidates: "
    f"{s3_stats['candidate_pairs']:,}"
)

print(
    f"\nS2 largest block: "
    f"{s2_stats['largest_block']:,}"
)

print(
    f"S3 largest block: "
    f"{s3_stats['largest_block']:,}"
)

print(
    f"\nS2 overflow blocks: "
    f"{s2_stats['overflow_blocks']:,}"
)

print(
    f"S3 overflow blocks: "
    f"{s3_stats['overflow_blocks']:,}"
)


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)

result = pl.DataFrame([
    {
        "strategy":
            "country_name_token_address_number",

        "currently_missed_pairs":
            missed_pairs.height,

        "recovered_pairs":
            total_recovered,

        "recovery_percent":
            recovery_percent,

        "estimated_candidate_pairs":
            total_candidates,

        "candidates_per_recovered_pair":
            candidates_per_recovered,

        "s2_candidates":
            s2_stats["candidate_pairs"],

        "s3_candidates":
            s3_stats["candidate_pairs"],

        "s2_largest_block":
            s2_stats["largest_block"],

        "s3_largest_block":
            s3_stats["largest_block"],

        "s2_overflow_blocks":
            s2_stats["overflow_blocks"],

        "s3_overflow_blocks":
            s3_stats["overflow_blocks"],
    }
])

result.write_csv(
    OUTPUT_FILE,
    separator="\t",
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
