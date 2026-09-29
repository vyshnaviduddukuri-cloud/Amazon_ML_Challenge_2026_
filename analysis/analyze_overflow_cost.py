import os
import polars as pl

# ============================================================
# OVERFLOW STRATEGY CANDIDATE-COST SHADOW EXPERIMENT
# ============================================================
#
# This experiment DOES NOT modify build_candidates_5.py.
#
# It measures the estimated candidate cost of alternative
# overflow blocking strategies on the S1 entities affected
# by the current block-cap problem.
#
# Strategies:
#
#   1. current_prefix4_core
#   2. address_token_core
#   3. address_token2_core
#
# We compare:
#
#   - estimated candidate pairs
#   - largest block
#   - number of overflow blocks
#   - recovered true pairs
#   - candidates per recovered true pair
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

TRAIN_DIR = "dataset/normalized"

MISSED_DIR = (
    "dataset/candidates/missed_analysis"
)

DIAGNOSTICS_FILE = os.path.join(
    MISSED_DIR,
    "block_cap_diagnostics.parquet"
)

OUTPUT_FILE = os.path.join(
    MISSED_DIR,
    "overflow_cost_analysis.tsv"
)

PAIR_CAP = 5_000


# ============================================================
# ADDRESS STOPWORDS
# ============================================================

ADDRESS_STOPWORDS = {
    "road", "rd",
    "street", "st",
    "avenue", "ave",
    "near", "nr",
    "main",
    "north", "south",
    "east", "west",
    "floor", "fl",
    "suite", "ste",
    "apartment", "apt",
    "colony", "nagar",
    "rue", "place", "pl",
    "saint", "sainte",
    "cedex",
    "and", "of",
}


# ============================================================
# FILE HELPER
# ============================================================

def scan_normalized(path):
    return pl.scan_csv(
        path,
        separator="\t",
        infer_schema_length=1000,
        null_values=[""],
    )


# ============================================================
# LOAD DIAGNOSTICS
# ============================================================

print("=" * 70)
print("OVERFLOW STRATEGY CANDIDATE-COST SHADOW EXPERIMENT")
print("=" * 70)

print("\nLoading cap diagnostics...")

diagnostics = pl.read_parquet(
    DIAGNOSTICS_FILE
)

print(
    f"Diagnostic rows: {diagnostics.height:,}"
)

print(
    f"Diagnostic columns: {len(diagnostics.columns)}"
)


# ============================================================
# VERIFY EXPECTED COLUMNS
# ============================================================

required_columns = [
    "s1_id",
    "match_id",
    "match_source",
    "name_block_over_cap",
    "name_core_block_over_cap",
    "exact_name_missing_from_block",
    "exact_core_missing_from_block",
    "strong_cap_evidence",
]

missing = [
    c
    for c in required_columns
    if c not in diagnostics.columns
]

if missing:
    raise RuntimeError(
        "Missing expected diagnostic columns: "
        + ", ".join(missing)
    )


# ============================================================
# CAP-RELATED MISSED PAIRS
# ============================================================
#
# We specifically select the rows where the diagnostics indicate
# that the missed true pair is related to an exact-name or
# name-core block exceeding the cap.
#
# This avoids treating all 1.57M misses as overflow misses.
# ============================================================

cap_misses = diagnostics.filter(
    (
        pl.col("exact_name_missing_from_block")
        == True
    )
    |
    (
        pl.col("exact_core_missing_from_block")
        == True
    )
)

print(
    "\nCap-related missed true-pair rows: "
    f"{cap_misses.height:,}"
)


# ============================================================
# UNIQUE S1 IDS
# ============================================================

cap_s1_ids = (
    cap_misses
    .select("s1_id")
    .unique()
)

print(
    "Unique S1 entities affected: "
    f"{cap_s1_ids.height:,}"
)


# ============================================================
# LOAD NORMALIZED SOURCES
# ============================================================

print("\nLoading normalized source files...")

s1_path = os.path.join(
    TRAIN_DIR,
    "train_source1.tsv"
)

s2_path = os.path.join(
    TRAIN_DIR,
    "train_source2.tsv"
)

s3_path = os.path.join(
    TRAIN_DIR,
    "train_source3.tsv"
)

s1 = scan_normalized(s1_path)
s2 = scan_normalized(s2_path)
s3 = scan_normalized(s3_path)


# ============================================================
# TARGET S1
# ============================================================

print(
    "\nRestricting S1 to cap-related entities..."
)

target_ids = cap_s1_ids.get_column(
    "s1_id"
).to_list()

s1_target = (
    s1
    .filter(
        pl.col("entity_id").is_in(target_ids)
    )
    .select([
        "entity_id",
        "normalized_country",
        "normalized_name_core",
        "normalized_address",
    ])
)

print(
    "Target S1 rows:"
)

print(
    s1_target.select(
        pl.len()
    ).collect().item()
)


# ============================================================
# STRATEGY 1
# CURRENT PREFIX4
# ============================================================

def make_prefix4_keys(lf):

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
        .with_columns(
            pl.col("normalized_address")
            .fill_null("")
            .str.slice(0, 4)
            .alias("overflow_key")
        )
        .filter(
            pl.col("overflow_key") != ""
        )
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "overflow_key",
        ])
        .unique()
    )


# ============================================================
# STRATEGY 2
# ADDRESS TOKEN + NAME CORE
# ============================================================

def make_address_token_keys(lf):

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
        .with_columns(
            pl.col("normalized_address")
            .fill_null("")
            .str.split(" ")
        )
        .explode("normalized_address")
        .rename({
            "normalized_address": "address_token"
        })
        .filter(
            pl.col("address_token") != ""
        )
        .filter(
            ~pl.col("address_token")
            .is_in(ADDRESS_STOPWORDS)
        )
        .filter(
            pl.col("address_token")
            .str.len_chars() >= 2
        )
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "address_token",
        ])
        .unique()
    )


# ============================================================
# STRATEGY 3
# TWO ADDRESS TOKENS + NAME CORE
# ============================================================

def make_address_token2_keys(lf):

    tokens = make_address_token_keys(lf)

    a = tokens.rename({
        "address_token": "token_a"
    })

    b = tokens.rename({
        "address_token": "token_b"
    })

    pairs = (
        a
        .join(
            b,
            on=[
                "entity_id",
                "normalized_country",
                "normalized_name_core",
            ],
            how="inner",
        )
        .filter(
            pl.col("token_a")
            < pl.col("token_b")
        )
        .with_columns(
            (
                pl.col("token_a")
                + "\x1f"
                + pl.col("token_b")
            ).alias("overflow_key")
        )
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "overflow_key",
        ])
        .unique()
    )

    return pairs


# ============================================================
# COST CALCULATOR
# ============================================================

def calculate_cost(
    s1_keys,
    other_keys,
    key_columns,
    strategy,
    source,
):

    print(
        f"\nCalculating {strategy} "
        f"against {source}..."
    )

    s1_counts = (
        s1_keys
        .group_by(key_columns)
        .agg(
            pl.len().alias("s1_count")
        )
    )

    other_counts = (
        other_keys
        .group_by(key_columns)
        .agg(
            pl.len().alias("other_count")
        )
    )

    joined = (
        s1_counts
        .join(
            other_counts,
            on=key_columns,
            how="inner",
        )
        .with_columns(
            (
                pl.col("s1_count")
                *
                pl.col("other_count")
            ).alias("pair_count")
        )
    )

    stats = (
        joined
        .select([
            pl.len().alias(
                "num_blocks"
            ),

            pl.col("pair_count")
            .sum()
            .alias(
                "estimated_pairs"
            ),

            pl.col("pair_count")
            .max()
            .alias(
                "largest_block"
            ),

            pl.col("pair_count")
            .mean()
            .alias(
                "mean_block"
            ),

            pl.col("pair_count")
            .median()
            .alias(
                "median_block"
            ),

            (
                pl.col("pair_count")
                > PAIR_CAP
            )
            .sum()
            .alias(
                "overflow_blocks"
            ),
        ])
        .collect()
        .row(0, named=True)
    )

    print(
        f"Blocks:              "
        f"{stats['num_blocks']:,}"
    )

    print(
        f"Estimated candidates:"
        f" {stats['estimated_pairs']:,}"
    )

    print(
        f"Largest block:       "
        f"{stats['largest_block']:,}"
    )

    print(
        f"Mean block:          "
        f"{stats['mean_block']:,.2f}"
    )

    print(
        f"Median block:        "
        f"{stats['median_block']:,.2f}"
    )

    print(
        f"Overflow blocks:     "
        f"{stats['overflow_blocks']:,}"
    )

    return {
        "strategy": strategy,
        "source": source,
        "num_blocks": stats["num_blocks"],
        "estimated_pairs": stats[
            "estimated_pairs"
        ],
        "largest_block": stats[
            "largest_block"
        ],
        "mean_block": stats[
            "mean_block"
        ],
        "median_block": stats[
            "median_block"
        ],
        "overflow_blocks": stats[
            "overflow_blocks"
        ],
    }


# ============================================================
# BUILD STRATEGY KEYS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING OVERFLOW KEYS")
print("=" * 70)

print("\nCurrent prefix4 keys...")

s1_prefix = make_prefix4_keys(
    s1_target
)

s2_prefix = make_prefix4_keys(
    s2
)

s3_prefix = make_prefix4_keys(
    s3
)


print("\nAddress-token keys...")

s1_token = make_address_token_keys(
    s1_target
)

s2_token = make_address_token_keys(
    s2
)

s3_token = make_address_token_keys(
    s3
)


print("\nTwo-token keys...")

s1_token2 = make_address_token2_keys(
    s1_target
)

s2_token2 = make_address_token2_keys(
    s2
)

s3_token2 = make_address_token2_keys(
    s3
)


# ============================================================
# RUN COST EXPERIMENT
# ============================================================

results = []


# ------------------------------------------------------------
# PREFIX4
# ------------------------------------------------------------

results.append(
    calculate_cost(
        s1_prefix,
        s2_prefix,
        [
            "normalized_country",
            "normalized_name_core",
            "overflow_key",
        ],
        "current_prefix4_core",
        "S2",
    )
)

results.append(
    calculate_cost(
        s1_prefix,
        s3_prefix,
        [
            "normalized_country",
            "normalized_name_core",
            "overflow_key",
        ],
        "current_prefix4_core",
        "S3",
    )
)


# ------------------------------------------------------------
# ADDRESS TOKEN
# ------------------------------------------------------------

results.append(
    calculate_cost(
        s1_token,
        s2_token,
        [
            "normalized_country",
            "normalized_name_core",
            "address_token",
        ],
        "address_token_core",
        "S2",
    )
)

results.append(
    calculate_cost(
        s1_token,
        s3_token,
        [
            "normalized_country",
            "normalized_name_core",
            "address_token",
        ],
        "address_token_core",
        "S3",
    )
)


# ------------------------------------------------------------
# TWO ADDRESS TOKENS
# ------------------------------------------------------------

results.append(
    calculate_cost(
        s1_token2,
        s2_token2,
        [
            "normalized_country",
            "normalized_name_core",
            "overflow_key",
        ],
        "address_token2_core",
        "S2",
    )
)

results.append(
    calculate_cost(
        s1_token2,
        s3_token2,
        [
            "normalized_country",
            "normalized_name_core",
            "overflow_key",
        ],
        "address_token2_core",
        "S3",
    )
)


# ============================================================
# COMBINE S2 + S3
# ============================================================

result_df = pl.DataFrame(results)

summary = (
    result_df
    .group_by("strategy")
    .agg([
        pl.col("estimated_pairs")
        .sum()
        .alias(
            "estimated_candidate_pairs"
        ),

        pl.col("num_blocks")
        .sum()
        .alias(
            "total_blocks"
        ),

        pl.col("largest_block")
        .max()
        .alias(
            "largest_block"
        ),

        pl.col("overflow_blocks")
        .sum()
        .alias(
            "overflow_blocks"
        ),
    ])
    .sort(
        "estimated_candidate_pairs"
    )
)


# ============================================================
# RECOVERY DATA
# ============================================================

RECOVERY = {
    "current_prefix4_core": 0,
    "address_token_core": 40_710,
    "address_token2_core": 36_386,
}


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CANDIDATE COST + RECOVERY SUMMARY")
print("=" * 70)

final_rows = []

for row in summary.iter_rows(
    named=True
):

    strategy = row["strategy"]

    recovered = RECOVERY.get(
        strategy,
        0
    )

    candidate_count = row[
        "estimated_candidate_pairs"
    ]

    if recovered > 0:
        cost_per_recovered = (
            candidate_count
            / recovered
        )
    else:
        cost_per_recovered = None

    final_rows.append({
        "strategy": strategy,

        "estimated_candidate_pairs":
            candidate_count,

        "recovered_true_pairs":
            recovered,

        "candidates_per_recovered_pair":
            cost_per_recovered,

        "largest_block":
            row["largest_block"],

        "overflow_blocks":
            row["overflow_blocks"],

        "total_blocks":
            row["total_blocks"],
    })

    print(
        f"\n{strategy}"
    )

    print(
        f"  Estimated candidates : "
        f"{candidate_count:,}"
    )

    print(
        f"  True pairs recovered : "
        f"{recovered:,}"
    )

    if cost_per_recovered is None:
        print(
            "  Candidates/recovered : N/A"
        )
    else:
        print(
            f"  Candidates/recovered : "
            f"{cost_per_recovered:,.2f}"
        )

    print(
        f"  Largest block        : "
        f"{row['largest_block']:,}"
    )

    print(
        f"  Overflow blocks      : "
        f"{row['overflow_blocks']:,}"
    )


# ============================================================
# SAVE
# ============================================================

final_df = pl.DataFrame(
    final_rows
)

final_df.write_csv(
    OUTPUT_FILE,
    separator="\t"
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE")
print("=" * 70)
