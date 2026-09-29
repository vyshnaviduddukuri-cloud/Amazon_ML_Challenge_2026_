import os
import polars as pl

# ============================================================
# HYBRID OVERFLOW SHADOW EXPERIMENT
# ============================================================
#
# Primary strategy:
#
#   country + name_core + informative_address_token
#
# If a block exceeds 5,000 candidate pairs:
#
#   country + name_core + token1 + token2
#
# The second-token strategy is generated ONLY for oversized
# blocks.
#
# This is a SHADOW experiment.
# It does NOT modify build_candidates_5.py.
# ============================================================

NORMALIZED_DIR = "dataset/normalized"
MISSED_DIR = "dataset/candidates/missed_analysis"

DIAGNOSTICS_FILE = os.path.join(
    MISSED_DIR,
    "block_cap_diagnostics.parquet"
)

OUTPUT_FILE = os.path.join(
    MISSED_DIR,
    "hybrid_overflow_analysis.tsv"
)

PAIR_CAP = 5_000

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
# LOADERS
# ============================================================

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


# ============================================================
# INFORMATIVE ADDRESS TOKENS
# ============================================================

def make_tokens(lf):
    """
    One row per entity + informative address token.
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
# LOAD DIAGNOSTICS
# ============================================================

print("=" * 70)
print("HYBRID OVERFLOW SHADOW EXPERIMENT")
print("=" * 70)

print("\nLoading diagnostics...")

diag = pl.read_parquet(
    DIAGNOSTICS_FILE
)

cap_misses = diag.filter(
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
    f"Cap-related missed pairs: "
    f"{cap_misses.height:,}"
)

cap_s1_ids = (
    cap_misses
    .select("s1_id")
    .unique()
)

print(
    f"Unique affected S1 entities: "
    f"{cap_s1_ids.height:,}"
)


# ============================================================
# LOAD SOURCES
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
# TARGET S1
# ============================================================

target_ids = cap_s1_ids.get_column(
    "s1_id"
).to_list()

s1_target = (
    s1
    .filter(
        pl.col("entity_id")
        .is_in(target_ids)
    )
    .select([
        "entity_id",
        "normalized_country",
        "normalized_name_core",
        "normalized_address",
    ])
)


# ============================================================
# CREATE ADDRESS TOKEN TABLES
# ============================================================

print("\nCreating informative address-token tables...")

s1_tokens = make_tokens(
    s1_target
)

s2_tokens = make_tokens(
    s2
)

s3_tokens = make_tokens(
    s3
)


# ============================================================
# STEP 1:
# FIND OVERSIZED TOKEN BLOCKS
# ============================================================

print("\n" + "=" * 70)
print("FINDING OVERSIZED ADDRESS-TOKEN BLOCKS")
print("=" * 70)


def find_oversized_blocks(
    s1_token_lf,
    other_token_lf,
    source_name,
):

    print(
        f"\nChecking {source_name}..."
    )

    key_cols = [
        "normalized_country",
        "normalized_name_core",
        "address_token",
    ]

    s1_counts = (
        s1_token_lf
        .group_by(key_cols)
        .agg(
            pl.len().alias("s1_count")
        )
    )

    other_counts = (
        other_token_lf
        .group_by(key_cols)
        .agg(
            pl.len().alias("other_count")
        )
    )

    blocks = (
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
        .filter(
            pl.col("pair_count")
            > PAIR_CAP
        )
        .collect()
    )

    print(
        f"Oversized blocks: "
        f"{blocks.height:,}"
    )

    if blocks.height:
        print(
            f"Largest block: "
            f"{blocks['pair_count'].max():,}"
        )

    return blocks


s2_oversized = find_oversized_blocks(
    s1_tokens,
    s2_tokens,
    "S2"
)

s3_oversized = find_oversized_blocks(
    s1_tokens,
    s3_tokens,
    "S3"
)


# ============================================================
# STEP 2:
# BUILD SECOND-TOKEN KEYS ONLY FOR OVERSIZED BLOCKS
# ============================================================

def build_second_token_candidates(
    s1_tokens_lf,
    other_tokens_lf,
    oversized_blocks,
    source_name,
):

    if oversized_blocks.height == 0:
        return None

    print(
        f"\nBuilding second-token keys "
        f"for {source_name} oversized blocks..."
    )

    key_cols = [
        "normalized_country",
        "normalized_name_core",
    ]

    # --------------------------------------------------------
    # Only cores/countries that actually have oversized blocks.
    # --------------------------------------------------------

    target_block_keys = (
        oversized_blocks
        .select(key_cols)
        .unique()
    )

    # --------------------------------------------------------
    # Restrict token tables to affected country + name_core.
    # --------------------------------------------------------

    s1_small = (
        s1_tokens_lf
        .join(
            target_block_keys.lazy(),
            on=key_cols,
            how="inner",
        )
    )

    other_small = (
        other_tokens_lf
        .join(
            target_block_keys.lazy(),
            on=key_cols,
            how="inner",
        )
    )

    # --------------------------------------------------------
    # For each entity, create token pairs.
    #
    # token_a < token_b makes unordered pairs unique.
    # --------------------------------------------------------

    s1_a = s1_small.rename({
        "address_token": "token_a"
    })

    s1_b = s1_small.rename({
        "address_token": "token_b"
    })

    s1_pairs = (
        s1_a
        .join(
            s1_b,
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
            ).alias("token_pair")
        )
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "token_pair",
        ])
        .unique()
    )

    other_a = other_small.rename({
        "address_token": "token_a"
    })

    other_b = other_small.rename({
        "address_token": "token_b"
    })

    other_pairs = (
        other_a
        .join(
            other_b,
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
            ).alias("token_pair")
        )
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "token_pair",
        ])
        .unique()
    )

    return (
        s1_pairs,
        other_pairs,
    )


s2_pairs = build_second_token_candidates(
    s1_tokens,
    s2_tokens,
    s2_oversized,
    "S2"
)

s3_pairs = build_second_token_candidates(
    s1_tokens,
    s3_tokens,
    s3_oversized,
    "S3"
)


# ============================================================
# STEP 3:
# CALCULATE SECOND-TOKEN COST
# ============================================================

def calculate_second_token_cost(
    pair_tables,
    source_name,
):

    if pair_tables is None:
        return {
            "source": source_name,
            "blocks": 0,
            "pairs": 0,
            "largest_block": 0,
            "overflow_blocks": 0,
        }

    s1_pairs, other_pairs = pair_tables

    key_cols = [
        "normalized_country",
        "normalized_name_core",
        "token_pair",
    ]

    print(
        f"\nCalculating hybrid cost "
        f"for {source_name}..."
    )

    s1_counts = (
        s1_pairs
        .group_by(key_cols)
        .agg(
            pl.len().alias("s1_count")
        )
    )

    other_counts = (
        other_pairs
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
            pl.len().alias(
                "blocks"
            ),

            pl.col("pair_count")
            .sum()
            .alias(
                "pairs"
            ),

            pl.col("pair_count")
            .max()
            .alias(
                "largest_block"
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
        f"{stats['blocks']:,}"
    )

    print(
        f"Estimated candidates:"
        f" {stats['pairs']:,}"
    )

    print(
        f"Largest block:       "
        f"{stats['largest_block']:,}"
    )

    print(
        f"Overflow blocks:     "
        f"{stats['overflow_blocks']:,}"
    )

    return {
        "source": source_name,
        **stats,
    }


s2_hybrid = calculate_second_token_cost(
    s2_pairs,
    "S2"
)

s3_hybrid = calculate_second_token_cost(
    s3_pairs,
    "S3"
)


# ============================================================
# STEP 4:
# TEST RECOVERY OF THE 45,304 CAP MISSES
# ============================================================

print("\n" + "=" * 70)
print("TESTING HYBRID RECOVERY")
print("=" * 70)


def recover_with_token_pair(
    cap_df,
    s1_pair_lf,
    other_pair_lf,
    source_name,
):

    source_misses = cap_df.filter(
        pl.col("match_source")
        == source_name
    )

    if source_misses.height == 0:
        return 0

    print(
        f"\nTesting recovery against {source_name}: "
        f"{source_misses.height:,} cap misses"
    )

    # --------------------------------------------------------
    # Get the true S1 and candidate IDs involved in the
    # cap-related missed pairs.
    # --------------------------------------------------------

    s1_ids = (
        source_misses
        .select("s1_id")
        .unique()
        .get_column("s1_id")
        .to_list()
    )

    match_ids = (
        source_misses
        .select("match_id")
        .unique()
        .get_column("match_id")
        .to_list()
    )

    # --------------------------------------------------------
    # The pair tables are LazyFrames.
    # Materialize only the small affected subsets needed for
    # this recovery test.
    # --------------------------------------------------------

    true_s1 = (
        s1_pair_lf
        .filter(
            pl.col("entity_id")
            .is_in(s1_ids)
        )
        .select([
            pl.col("entity_id")
            .alias("s1_id"),
            "normalized_country",
            "normalized_name_core",
            "token_pair",
        ])
        .collect()
    )

    true_other = (
        other_pair_lf
        .filter(
            pl.col("entity_id")
            .is_in(match_ids)
        )
        .select([
            pl.col("entity_id")
            .alias("match_id"),
            "normalized_country",
            "normalized_name_core",
            "token_pair",
        ])
        .collect()
    )

    # --------------------------------------------------------
    # A true pair is recoverable if:
    #
    # S1 entity
    #   and
    # true S2/S3 entity
    #
    # share the same:
    #   country
    #   name_core
    #   token_pair
    # --------------------------------------------------------

    matches = (
        source_misses
        .select([
            "s1_id",
            "match_id",
        ])
        .join(
            true_s1,
            on="s1_id",
            how="inner",
        )
        .join(
            true_other,
            on=[
                "match_id",
                "normalized_country",
                "normalized_name_core",
                "token_pair",
            ],
            how="inner",
        )
        .select([
            "s1_id",
            "match_id",
        ])
        .unique()
    )

    # matches is already a DataFrame because both joins above
    # are DataFrame joins.
    recovered = matches.height

    print(
        f"Recovered by second-token fallback: "
        f"{recovered:,}"
    )

    return recovered 
# ============================================================
# RUN RECOVERY TEST
# ============================================================

s2_recovered = recover_with_token_pair(
    cap_misses,
    s2_pairs[0] if s2_pairs else None,
    s2_pairs[1] if s2_pairs else None,
    "S2",
)

s3_recovered = recover_with_token_pair(
    cap_misses,
    s3_pairs[0] if s3_pairs else None,
    s3_pairs[1] if s3_pairs else None,
    "S3",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

total_recovered = (
    s2_recovered
    + s3_recovered
)

total_hybrid_candidates = (
    s2_hybrid["pairs"]
    + s3_hybrid["pairs"]
)

print("\n" + "=" * 70)
print("HYBRID OVERFLOW SUMMARY")
print("=" * 70)

print(
    f"\nCap-related missed pairs: "
    f"{cap_misses.height:,}"
)

print(
    f"Recovered by hybrid second-token fallback: "
    f"{total_recovered:,}"
)

if cap_misses.height:
    print(
        f"Recovery percentage: "
        f"{100 * total_recovered / cap_misses.height:.4f}%"
    )

print(
    f"\nEstimated hybrid candidate pairs: "
    f"{total_hybrid_candidates:,}"
)

if total_recovered:
    print(
        f"Candidates per recovered pair: "
        f"{total_hybrid_candidates / total_recovered:,.2f}"
    )

print(
    f"\nS2 hybrid candidates: "
    f"{s2_hybrid['pairs']:,}"
)

print(
    f"S3 hybrid candidates: "
    f"{s3_hybrid['pairs']:,}"
)

print(
    f"\nS2 largest hybrid block: "
    f"{s2_hybrid['largest_block']:,}"
)

print(
    f"S3 largest hybrid block: "
    f"{s3_hybrid['largest_block']:,}"
)

print(
    f"\nS2 remaining overflow blocks: "
    f"{s2_hybrid['overflow_blocks']:,}"
)

print(
    f"S3 remaining overflow blocks: "
    f"{s3_hybrid['overflow_blocks']:,}"
)


   
  


# ============================================================
# SAVE
# ============================================================

result = pl.DataFrame([
    {
        "strategy": "hybrid_address_token_then_token2",

        "cap_missed_pairs":
            cap_misses.height,

        "recovered_pairs":
            total_recovered,

        "recovery_percent":
            (
                100 * total_recovered
                / cap_misses.height
            ),

        "estimated_candidate_pairs":
            total_hybrid_candidates,

        "candidates_per_recovered_pair":
            (
                total_hybrid_candidates
                / total_recovered
                if total_recovered
                else None
            ),

        "s2_candidates":
            s2_hybrid["pairs"],

        "s3_candidates":
            s3_hybrid["pairs"],

        "s2_largest_block":
            s2_hybrid["largest_block"],

        "s3_largest_block":
            s3_hybrid["largest_block"],

        "s2_remaining_overflow":
            s2_hybrid["overflow_blocks"],

        "s3_remaining_overflow":
            s3_hybrid["overflow_blocks"],
    }
])

result.write_csv(
    OUTPUT_FILE,
    separator="\t"
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("HYBRID EXPERIMENT COMPLETE")
print("=" * 70)
