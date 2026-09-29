import polars as pl

STOP = {
    "road","rd","street","st","avenue","ave","near","nr","main",
    "north","south","east","west","floor","fl","suite","ste",
    "apartment","apt","colony","nagar","rue","place","pl",
    "saint","sainte","cedex","and","of","no","new"
}

missed = pl.read_parquet(
    "dataset/candidates/missed_true_pairs_current.parquet"
)

s1 = (
    pl.scan_parquet("dataset/candidates/slim_parquet/train_s1.parquet")
    .select(["entity_id", "normalized_address"])
    .rename({
        "entity_id": "source1_entity_id",
        "normalized_address": "s1_address"
    })
)

results = []

for source in ["s2", "s3"]:
    other = (
        pl.scan_parquet(
            f"dataset/candidates/slim_parquet/train_{source}.parquet"
        )
        .select(["entity_id", "normalized_address"])
        .rename({
            "entity_id": "candidate_entity_id",
            "normalized_address": "other_address"
        })
    )

    pairs = (
        missed.lazy()
        .join(s1, on="source1_entity_id")
        .join(other, on="candidate_entity_id")
        .filter(
            (pl.col("s1_address").str.len_chars() > 0) &
            (pl.col("other_address").str.len_chars() > 0)
        )
        .with_columns([
            pl.col("s1_address").str.split(" ").alias("s1_tokens"),
            pl.col("other_address").str.split(" ").alias("other_tokens")
        ])
        .with_columns(
            pl.col("s1_tokens")
            .list.set_intersection(pl.col("other_tokens"))
            .alias("shared")
        )
        .with_columns([
            pl.col("s1_address")
            .str.extract(r"(?i)(?:^| )([0-9]{1,6})(?: |$)", 1)
            .alias("s1_number"),
            pl.col("other_address")
            .str.extract(r"(?i)(?:^| )([0-9]{1,6})(?: |$)", 1)
            .alias("other_number")
        ])
        .filter(
            (pl.col("s1_number").is_not_null()) &
            (pl.col("s1_number") == pl.col("other_number"))
        )
        .with_columns(
            pl.col("shared")
            .list.eval(
                pl.element().filter(
                    ~pl.element().is_in(list(STOP))
                )
            )
            .alias("informative_shared")
        )
        .filter(pl.col("informative_shared").list.len() > 0)
        .select([
            "source1_entity_id",
            "candidate_entity_id",
            "s1_number",
            "informative_shared"
        ])
        .collect(engine="streaming")
    )

    pairs = pairs.with_columns(
        pl.lit(source.upper()).alias("source")
    )
    results.append(pairs)

all_pairs = pl.concat(results)

print(f"Missed pairs with same number + informative token: {all_pairs.height:,}")

unique_pairs = (
    all_pairs
    .select(["source1_entity_id", "candidate_entity_id", "source"])
    .unique()
)

print(f"Unique missed pairs: {unique_pairs.height:,}")

print()
print("Top informative tokens:")

token_freq = (
    all_pairs
    .explode("informative_shared")
    .group_by("informative_shared")
    .agg([
        pl.len().alias("missed_pairs"),
        pl.col("source1_entity_id").n_unique().alias("unique_s1")
    ])
    .sort("missed_pairs", descending=True)
)

print(token_freq.head(30))

unique_pairs.write_parquet(
    "dataset/block_audit/missed_one_token_number_pairs.parquet"
)

token_freq.write_csv(
    "dataset/block_audit/missed_one_token_number_frequency.tsv",
    separator="\t"
)

print()
print("Saved rescue analysis files.")
