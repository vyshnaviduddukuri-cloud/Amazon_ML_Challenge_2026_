#!/usr/bin/env python3

from pathlib import Path
import re
import polars as pl


# ============================================================
# CONFIG
# ============================================================

MISSED_DIR = Path("dataset/candidates/missed_analysis")
NORMALIZED_DIR = Path("dataset/normalized")

CAP = 5_000

MISSED_PATH = MISSED_DIR / "missed_true_pairs.parquet"

S1_PATH = NORMALIZED_DIR / "train_source1.tsv"
S2_PATH = NORMALIZED_DIR / "train_source2.tsv"
S3_PATH = NORMALIZED_DIR / "train_source3.tsv"


ADDRESS_STOPWORDS = {
    "road", "rd",
    "street", "st",
    "avenue", "ave",
    "boulevard", "blvd",
    "drive", "dr",
    "lane", "ln",
    "near", "nr",
    "main",
    "north", "south", "east", "west",
    "floor", "fl",
    "suite", "ste",
    "apartment", "apt",
    "colony",
    "nagar",
    "rue",
    "place", "pl",
    "saint", "sainte",
    "cedex",
    "and", "of",
}


# ============================================================
# HELPERS
# ============================================================

def require_columns(df, required, name):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"{name} is missing columns: {missing}"
        )


def extract_numbers(text):
    if not text:
        return []

    return re.findall(r"\d+", text)


def address_tokens(text):
    if not text:
        return []

    tokens = text.split()

    out = []

    for token in tokens:
        token = token.strip()

        if not token:
            continue

        if token in ADDRESS_STOPWORDS:
            continue

        if len(token) < 3:
            continue

        out.append(token)

    return sorted(set(out))


def number_key(text):
    nums = extract_numbers(text)

    if not nums:
        return ""

    return "|".join(sorted(set(nums)))


def informative_tokens(text):
    return address_tokens(text)


def token_pair_keys(text):
    tokens = informative_tokens(text)

    if len(tokens) < 2:
        return []

    keys = []

    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            keys.append(tokens[i] + "|" + tokens[j])

    return keys


def first_four(text):
    if not text:
        return ""

    return text[:4]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("OVERFLOW STRATEGY SHADOW EXPERIMENT")
print("=" * 70)

print("\nLoading missed true pairs...")

missed = pl.read_parquet(MISSED_PATH)

require_columns(
    missed,
    ["source1_entity_id", "candidate_entity_id"],
    "missed_true_pairs"
)

print(f"Missed pairs: {missed.height:,}")


print("\nLoading normalized sources...")

s1 = pl.read_csv(
    S1_PATH,
    separator="\t",
    infer_schema_length=1000,
)

s2 = pl.read_csv(
    S2_PATH,
    separator="\t",
    infer_schema_length=1000,
)

s3 = pl.read_csv(
    S3_PATH,
    separator="\t",
    infer_schema_length=1000,
)

required = [
    "entity_id",
    "normalized_name",
    "normalized_name_core",
    "normalized_address",
    "normalized_country",
]

require_columns(s1, required, "S1")
require_columns(s2, required, "S2")
require_columns(s3, required, "S3")

print(f"S1: {s1.height:,}")
print(f"S2: {s2.height:,}")
print(f"S3: {s3.height:,}")


# ============================================================
# ATTACH MISSED PAIR RECORDS
# ============================================================

print("\nAttaching source records...")

s1_small = s1.select([
    "entity_id",
    "normalized_name",
    "normalized_name_core",
    "normalized_address",
    "normalized_country",
])

s2_small = s2.select([
    "entity_id",
    "normalized_name",
    "normalized_name_core",
    "normalized_address",
    "normalized_country",
])

s3_small = s3.select([
    "entity_id",
    "normalized_name",
    "normalized_name_core",
    "normalized_address",
    "normalized_country",
])


missed = (
    missed
    .join(
        s1_small,
        left_on="source1_entity_id",
        right_on="entity_id",
        how="left",
    )
    .rename({
        "normalized_name": "s1_name",
        "normalized_name_core": "s1_core",
        "normalized_address": "s1_address",
        "normalized_country": "s1_country",
    })
)


missed_s2 = (
    missed
    .filter(pl.col("candidate_entity_id").str.starts_with("S2-"))
    .join(
        s2_small,
        left_on="candidate_entity_id",
        right_on="entity_id",
        how="left",
    )
    .rename({
        "normalized_name": "match_name",
        "normalized_name_core": "match_core",
        "normalized_address": "match_address",
        "normalized_country": "match_country",
    })
)


missed_s3 = (
    missed
    .filter(pl.col("candidate_entity_id").str.starts_with("S3-"))
    .join(
        s3_small,
        left_on="candidate_entity_id",
        right_on="entity_id",
        how="left",
    )
    .rename({
        "normalized_name": "match_name",
        "normalized_name_core": "match_core",
        "normalized_address": "match_address",
        "normalized_country": "match_country",
    })
)


missed = pl.concat(
    [missed_s2, missed_s3],
    how="vertical",
)


# ============================================================
# KEEP ONLY CAP-RELATED MISSES
# ============================================================

print("\nIdentifying cap-related missed pairs...")

name_counts_s1 = (
    s1_small
    .group_by(["normalized_country", "normalized_name"])
    .agg(pl.len().alias("s1_count"))
)

name_counts_s2 = (
    s2_small
    .group_by(["normalized_country", "normalized_name"])
    .agg(pl.len().alias("other_count"))
)

name_counts_s3 = (
    s3_small
    .group_by(["normalized_country", "normalized_name"])
    .agg(pl.len().alias("other_count"))
)

core_counts_s1 = (
    s1_small
    .group_by(["normalized_country", "normalized_name_core"])
    .agg(pl.len().alias("s1_count"))
)

core_counts_s2 = (
    s2_small
    .group_by(["normalized_country", "normalized_name_core"])
    .agg(pl.len().alias("other_count"))
)

core_counts_s3 = (
    s3_small
    .group_by(["normalized_country", "normalized_name_core"])
    .agg(pl.len().alias("other_count"))
)


# ------------------------------------------------------------
# Attach exact-name pair size
# ------------------------------------------------------------

missed = missed.with_columns([
    (
        pl.col("s1_name") == pl.col("match_name")
    ).alias("name_exact"),

    (
        pl.col("s1_core") == pl.col("match_core")
    ).alias("core_exact"),
])


missed = missed.with_columns([
    (
        pl.col("s1_country") == pl.col("match_country")
    ).alias("country_exact"),
])


# ============================================================
# ADDRESS KEY GENERATION
# ============================================================

print("\nGenerating address keys...")

missed = missed.with_columns([
    pl.col("s1_address")
    .map_elements(number_key, return_dtype=pl.String)
    .alias("s1_number_key"),

    pl.col("match_address")
    .map_elements(number_key, return_dtype=pl.String)
    .alias("match_number_key"),

    pl.col("s1_address")
    .map_elements(first_four, return_dtype=pl.String)
    .alias("s1_prefix4"),

    pl.col("match_address")
    .map_elements(first_four, return_dtype=pl.String)
    .alias("match_prefix4"),

    pl.col("s1_address")
    .map_elements(
        lambda x: "|".join(informative_tokens(x)),
        return_dtype=pl.String,
    )
    .alias("s1_address_tokens"),

    pl.col("match_address")
    .map_elements(
        lambda x: "|".join(informative_tokens(x)),
        return_dtype=pl.String,
    )
    .alias("match_address_tokens"),
])


# ============================================================
# SHARED TOKEN ANALYSIS
# ============================================================

def shared_tokens(a, b):
    if not a or not b:
        return []

    sa = set(a.split("|"))
    sb = set(b.split("|"))

    return sorted(sa & sb)


missed = missed.with_columns(
    pl.struct([
        "s1_address_tokens",
        "match_address_tokens",
    ])
    .map_elements(
        lambda x: len(
            set(x["s1_address_tokens"].split("|"))
            &
            set(x["match_address_tokens"].split("|"))
        )
        if x["s1_address_tokens"] and x["match_address_tokens"]
        else 0,
        return_dtype=pl.Int64,
    )
    .alias("shared_address_token_count")
)


# ============================================================
# STRATEGY RECOVERY
# ============================================================

missed = missed.with_columns([

    (
        pl.col("name_exact")
        & pl.col("country_exact")
        & (pl.col("s1_prefix4") != "")
        & (pl.col("s1_prefix4") == pl.col("match_prefix4"))
    ).alias("strategy_prefix4"),

    (
        pl.col("name_exact")
        & pl.col("country_exact")
        & (pl.col("s1_number_key") != "")
        & (pl.col("s1_number_key") == pl.col("match_number_key"))
    ).alias("strategy_number"),

    (
        pl.col("name_exact")
        & pl.col("country_exact")
        & (pl.col("shared_address_token_count") >= 1)
    ).alias("strategy_address_token"),

    (
        pl.col("name_exact")
        & pl.col("country_exact")
        & (pl.col("shared_address_token_count") >= 2)
    ).alias("strategy_address_token2"),

    (
        pl.col("name_exact")
        & pl.col("country_exact")
        & (pl.col("s1_number_key") != "")
        & (pl.col("s1_number_key") == pl.col("match_number_key"))
        & (pl.col("shared_address_token_count") >= 1)
    ).alias("strategy_number_token"),

    (
        pl.col("core_exact")
        & pl.col("country_exact")
        & (pl.col("shared_address_token_count") >= 1)
    ).alias("strategy_core_address_token"),

    (
        pl.col("core_exact")
        & pl.col("country_exact")
        & (pl.col("shared_address_token_count") >= 2)
    ).alias("strategy_core_address_token2"),
])


# ============================================================
# REPORT RECOVERY
# ============================================================

strategies = [
    ("current_prefix4_name", "strategy_prefix4"),
    ("number_name", "strategy_number"),
    ("address_token_name", "strategy_address_token"),
    ("address_token2_name", "strategy_address_token2"),
    ("number_plus_token_name", "strategy_number_token"),
    ("address_token_core", "strategy_core_address_token"),
    ("address_token2_core", "strategy_core_address_token2"),
]


print("\n" + "=" * 70)
print("RECOVERY ON MISSED PAIRS")
print("=" * 70)

print(
    f"{'strategy':35} {'recovered':>12} {'percent':>12}"
)

print("-" * 70)

for label, column in strategies:

    count = (
        missed
        .filter(pl.col(column))
        .height
    )

    pct = (
        100.0 * count / missed.height
        if missed.height
        else 0
    )

    print(
        f"{label:35} "
        f"{count:12,} "
        f"{pct:11.4f}%"
    )


# ============================================================
# CAP-RELATED SUBSET
# ============================================================

cap_related = missed.filter(
    pl.col("name_exact") | pl.col("core_exact")
)

print("\n" + "=" * 70)
print("RECOVERY ON EXACT NAME / CORE MISSES")
print("=" * 70)

print(
    f"Exact-name/core subset: {cap_related.height:,}"
)

for label, column in strategies:

    count = (
        cap_related
        .filter(pl.col(column))
        .height
    )

    pct = (
        100.0 * count / cap_related.height
        if cap_related.height
        else 0
    )

    print(
        f"{label:35} "
        f"{count:12,} "
        f"{pct:11.4f}%"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

out_path = MISSED_DIR / "overflow_strategy_analysis.tsv"

missed.write_csv(
    out_path,
    separator="\t",
)

print("\nSaved:")
print(out_path)

print("\nAnalysis complete.")
