import polars as pl
from pathlib import Path

missed = pl.read_parquet(
    "dataset/candidates/missed_true_pairs_current.parquet"
)

samples = []

for source in ["s2", "s3"]:

    s1 = (
        pl.scan_parquet(
            f"dataset/candidates/slim_parquet/train_s1.parquet"
        )
        .select([
            "entity_id",
            "normalized_address",
            "normalized_name"
        ])
        .rename({
            "entity_id": "source1_entity_id",
            "normalized_address": "s1_address",
            "normalized_name": "s1_name"
        })
    )

    other = (
        pl.scan_parquet(
            f"dataset/candidates/slim_parquet/train_{source}.parquet"
        )
        .select([
            "entity_id",
            "normalized_address",
            "normalized_name"
        ])
        .rename({
            "entity_id": "candidate_entity_id",
            "normalized_address": "other_address",
            "normalized_name": "other_name"
        })
    )

    pairs = (
        missed.lazy()
        .join(s1, on="source1_entity_id")
        .join(other, on="candidate_entity_id")
        .filter(
            pl.col("s1_address").str.len_chars() > 0
        )
        .filter(
            pl.col("other_address").str.len_chars() > 0
        )
        .with_columns([
            pl.col("s1_address").str.split(" ").alias("s1_tokens"),
            pl.col("other_address").str.split(" ").alias("other_tokens")
        ])
        .with_columns(
            pl.col("s1_tokens")
            .list.set_intersection(pl.col("other_tokens"))
            .alias("shared_tokens")
        )
        .explode("shared_tokens")
        .filter(pl.col("shared_tokens") != "")
        .select([
            "source1_entity_id",
            "candidate_entity_id",
            "shared_tokens"
        ])
        .collect(engine="streaming")
    )

    pairs = pairs.with_columns(
        pl.lit(source.upper()).alias("source")
    )

    samples.append(pairs)

all_pairs = pl.concat(samples)

print(f"Missed/shared-address-token rows: {all_pairs.height:,}")

freq = (
    all_pairs
    .group_by("shared_tokens")
    .agg([
        pl.len().alias("missed_true_pairs"),
        pl.col("source1_entity_id").n_unique().alias("unique_s1")
    ])
    .sort("missed_true_pairs", descending=True)
)

print()
print(freq.head(40))

freq.write_csv(
    "dataset/block_audit/missed_address_token_frequency.tsv",
    separator="\t"
)

print()
print("Saved: dataset/block_audit/missed_address_token_frequency.tsv")
