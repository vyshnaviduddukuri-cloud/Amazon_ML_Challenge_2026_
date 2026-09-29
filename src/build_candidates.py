

import argparse
import gc
from pathlib import Path

import polars as pl



SEP = "\x1f"

NAME_STOPWORDS = {
    "the", "and", "of", "for", "new", "inc", "llc", "ltd", "co", "company",
    "corporation", "corp", "group", "international", "intl", "private",
    "limited", "pvt", "enterprises", "enterprise", "sarl", "sas", "sa",
}

ADDRESS_STOPWORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "near", "nr", "main",
    "north", "south", "east", "west", "floor", "fl", "suite", "ste",
    "apartment", "apt", "colony", "nagar", "rue", "place", "pl", "saint",
    "sainte", "cedex", "and", "of",
}

MIN_TOKEN_LEN = 3

MAX_PAIRS_PER_BLOCK_KEY = 5_000

NUM_CANDIDATE_PARTITIONS = 64

SLIM_COLUMNS = [
    "entity_id",
    "normalized_name",
    "normalized_name_core",
    "normalized_address",
    "normalized_country",
]

CORE_BLOCK_KEYS = {
    "name_exact": "normalized_name",
    "name_core": "normalized_name_core",
    "address_exact": "normalized_address",
}



def ensure_slim_parquet(
    tsv_path: Path,
    parquet_path: Path
) -> Path:

    if (
        parquet_path.exists()
        and parquet_path.stat().st_mtime >= tsv_path.stat().st_mtime
    ):
        print(f"[reuse] {parquet_path}")
        return parquet_path

    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[slim] Creating {parquet_path} "
        f"from {tsv_path}"
    )

    (
        pl.scan_csv(
            tsv_path,
            separator="\t"
        )
        .select(SLIM_COLUMNS)
        .sink_parquet(parquet_path)
    )

    return parquet_path




def _composite_key(
    lf: pl.LazyFrame,
    key_col: str
) -> pl.LazyFrame:

    return lf.with_columns(
        (
            pl.col("normalized_country").fill_null("")
            + pl.lit(SEP)
            + pl.col(key_col)
        ).alias("_key")
    )




def generate_exact_with_fallback(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    key_col: str,
    block_label: str,
    other_label: str,
    max_pairs_per_key: int = MAX_PAIRS_PER_BLOCK_KEY,
) -> pl.LazyFrame:

    s1_k = _composite_key(
        s1_lf,
        key_col
    ).filter(
        (pl.col(key_col) != "")
        & pl.col(key_col).is_not_null()
    )

    other_k = _composite_key(
        other_lf,
        key_col
    ).filter(
        (pl.col(key_col) != "")
        & pl.col(key_col).is_not_null()
    )

    sizes_s1 = (
        s1_k
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n_s1")
        )
    )

    sizes_other = (
        other_k
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n_other")
        )
    )

    estimates = (
        sizes_s1
        .join(
            sizes_other,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n_s1")
                * pl.col("_n_other")
            ).alias("_pair_estimate")
        )
    )

    safe_keys = (
        estimates
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select("_key")
    )

    safe_pairs = (
        s1_k
        .join(
            safe_keys,
            on="_key",
            how="inner"
        )
        .select([
            "entity_id",
            "_key"
        ])
        .join(
            other_k
            .join(
                safe_keys,
                on="_key",
                how="inner"
            )
            .select([
                "entity_id",
                "_key"
            ]),
            on="_key",
            how="inner",
            suffix="_other",
        )
    )

    overflow_keys = (
        estimates
        .filter(
            pl.col("_pair_estimate")
            > max_pairs_per_key
        )
        .select("_key")
    )

    s1_overflow = (
        s1_k
        .join(
            overflow_keys,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_key")
                + pl.lit(SEP)
                + pl.col("normalized_address")
                .fill_null("")
                .str.slice(0, 4)
            ).alias("_compound_key")
        )
        .filter(
            pl.col("normalized_address") != ""
        )
    )

    other_overflow = (
        other_k
        .join(
            overflow_keys,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_key")
                + pl.lit(SEP)
                + pl.col("normalized_address")
                .fill_null("")
                .str.slice(0, 4)
            ).alias("_compound_key")
        )
        .filter(
            pl.col("normalized_address") != ""
        )
    )

    compound_pairs = (
        s1_overflow
        .select([
            "entity_id",
            "_compound_key"
        ])
        .join(
            other_overflow
            .select([
                "entity_id",
                "_compound_key"
            ]),
            on="_compound_key",
            how="inner",
            suffix="_other",
        )
    )

    return (
        pl.concat(
            [
                safe_pairs,
                compound_pairs
            ],
            how="vertical"
        )
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .with_columns(
            pl.lit(block_label)
            .alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )



def _tokenize(
    lf: pl.LazyFrame,
    text_col: str,
    stopwords: set
) -> pl.LazyFrame:

    return (
        lf
        .select([
            "entity_id",
            "normalized_country",
            text_col
        ])
        .filter(
            pl.col(text_col) != ""
        )
        .with_columns(
            pl.col(text_col)
            .str.split(" ")
            .alias("_tok")
        )
        .explode("_tok")
        .filter(
            pl.col("_tok")
            .str.len_chars()
            >= MIN_TOKEN_LEN
        )
        .filter(
            ~pl.col("_tok")
            .is_in(list(stopwords))
        )
        .unique(
            subset=[
                "entity_id",
                "normalized_country",
                "_tok"
            ]
        )
    )


def _token_frequencies(
    tok_lf: pl.LazyFrame
) -> pl.LazyFrame:

    return (
        tok_lf
        .group_by([
            "normalized_country",
            "_tok"
        ])
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_freq")
        )
    )



def generate_token_block_pairs(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    text_col: str,
    stopwords: set,
    block_label: str,
    other_label: str,
    max_pairs_per_token: int = MAX_PAIRS_PER_BLOCK_KEY,
) -> pl.LazyFrame:

    s1_tok = _tokenize(
        s1_lf,
        text_col,
        stopwords
    )

    other_tok = _tokenize(
        other_lf,
        text_col,
        stopwords
    )

    freq_s1 = (
        _token_frequencies(s1_tok)
        .rename({
            "_freq": "_freq_s1"
        })
    )

    freq_other = (
        _token_frequencies(other_tok)
        .rename({
            "_freq": "_freq_other"
        })
    )

    safe_tokens = (
        freq_s1
        .join(
            freq_other,
            on=[
                "normalized_country",
                "_tok"
            ],
            how="inner"
        )
        .with_columns(
            (
                pl.col("_freq_s1")
                * pl.col("_freq_other")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_token
        )
        .select([
            "normalized_country",
            "_tok"
        ])
    )

    s1_tok_f = (
        s1_tok
        .join(
            safe_tokens,
            on=[
                "normalized_country",
                "_tok"
            ],
            how="inner"
        )
    )

    other_tok_f = (
        other_tok
        .join(
            safe_tokens,
            on=[
                "normalized_country",
                "_tok"
            ],
            how="inner"
        )
    )

    joined = (
        s1_tok_f
        .select([
            "entity_id",
            "normalized_country",
            "_tok"
        ])
        .join(
            other_tok_f
            .select([
                "entity_id",
                "normalized_country",
                "_tok"
            ]),
            on=[
                "normalized_country",
                "_tok"
            ],
            how="inner",
            suffix="_other",
        )
    )

    return (
        joined
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .unique()
        .with_columns(
            pl.lit(block_label)
            .alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )




def generate_prefix_block_pairs(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    prefix_len: int = 4,
    max_pairs_per_key: int = 3_000,
) -> pl.LazyFrame:

    s1_p = (
        s1_lf
        .filter(
            pl.col("normalized_name_core")
            .is_not_null()
            & (
                pl.col("normalized_name_core")
                .str.len_chars()
                >= prefix_len
            )
        )
        .with_columns(
            (
                pl.col("normalized_country")
                .fill_null("")
                + pl.lit(SEP)
                + pl.col("normalized_name_core")
                .str.slice(0, prefix_len)
            ).alias("_key")
        )
    )

    other_p = (
        other_lf
        .filter(
            pl.col("normalized_name_core")
            .is_not_null()
            & (
                pl.col("normalized_name_core")
                .str.len_chars()
                >= prefix_len
            )
        )
        .with_columns(
            (
                pl.col("normalized_country")
                .fill_null("")
                + pl.lit(SEP)
                + pl.col("normalized_name_core")
                .str.slice(0, prefix_len)
            ).alias("_key")
        )
    )

    sizes_s1 = (
        s1_p
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n_s1")
        )
    )

    sizes_other = (
        other_p
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n_other")
        )
    )

    safe_keys = (
        sizes_s1
        .join(
            sizes_other,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n_s1")
                * pl.col("_n_other")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select("_key")
    )

    joined = (
        s1_p
        .join(
            safe_keys,
            on="_key",
            how="inner"
        )
        .select([
            "entity_id",
            "_key"
        ])
        .join(
            other_p
            .join(
                safe_keys,
                on="_key",
                how="inner"
            )
            .select([
                "entity_id",
                "_key"
            ]),
            on="_key",
            how="inner",
            suffix="_other",
        )
    )

    return (
        joined
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .unique()
        .with_columns(
            pl.lit(
                f"prefix_{prefix_len}"
            ).alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )




def generate_rare_token_number_blocks(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    max_token_freq: int = 2500,
    max_pairs_per_key: int = MAX_PAIRS_PER_BLOCK_KEY,
) -> pl.LazyFrame:

    s1_base = (
        s1_lf
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "normalized_address",
        ])
        .with_columns(
            pl.col("normalized_address")
            .str.extract(
                r"(?:^|[^0-9])([0-9]{1,6})(?:[^0-9]|$)",
                group_index=1
            )
            .fill_null("")
            .alias("address_num")
        )
        .filter(
            (pl.col("normalized_country") != "")
            & (pl.col("address_num") != "")
            & (pl.col("normalized_name_core") != "")
        )
    )

    other_base = (
        other_lf
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core",
            "normalized_address",
        ])
        .with_columns(
            pl.col("normalized_address")
            .str.extract(
                r"(?:^|[^0-9])([0-9]{1,6})(?:[^0-9]|$)",
                group_index=1
            )
            .fill_null("")
            .alias("address_num")
        )
        .filter(
            (pl.col("normalized_country") != "")
            & (pl.col("address_num") != "")
            & (pl.col("normalized_name_core") != "")
        )
    )

    s1_tok = (
        s1_base
        .select([
            "entity_id",
            "normalized_country",
            "address_num",
            pl.col("normalized_name_core")
            .str.split(" ")
            .alias("tok")
        ])
        .explode("tok")
        .with_columns(
            pl.col("tok")
            .str.strip_chars()
            .alias("tok")
        )
        .filter(
            (
                pl.col("tok")
                .str.len_chars()
                >= MIN_TOKEN_LEN
            )
            & (
                ~pl.col("tok")
                .is_in(list(NAME_STOPWORDS))
            )
        )
        .unique(
            subset=[
                "entity_id",
                "normalized_country",
                "tok"
            ]
        )
    )

    other_tok = (
        other_base
        .select([
            "entity_id",
            "normalized_country",
            "address_num",
            pl.col("normalized_name_core")
            .str.split(" ")
            .alias("tok")
        ])
        .explode("tok")
        .with_columns(
            pl.col("tok")
            .str.strip_chars()
            .alias("tok")
        )
        .filter(
            (
                pl.col("tok")
                .str.len_chars()
                >= MIN_TOKEN_LEN
            )
            & (
                ~pl.col("tok")
                .is_in(list(NAME_STOPWORDS))
            )
        )
        .unique(
            subset=[
                "entity_id",
                "normalized_country",
                "tok"
            ]
        )
    )

    s1_freq = (
        s1_tok
        .group_by([
            "normalized_country",
            "tok"
        ])
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("freq")
        )
        .filter(
            pl.col("freq")
            <= max_token_freq
        )
    )

    other_freq = (
        other_tok
        .group_by([
            "normalized_country",
            "tok"
        ])
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("freq")
        )
        .filter(
            pl.col("freq")
            <= max_token_freq
        )
    )

    s1_rare = (
        s1_tok
        .join(
            s1_freq.select([
                "normalized_country",
                "tok"
            ]),
            on=[
                "normalized_country",
                "tok"
            ],
            how="inner"
        )
        .with_columns(
            (
                pl.col("normalized_country")
                + pl.lit(SEP)
                + pl.col("tok")
                + pl.lit(SEP)
                + pl.col("address_num")
            ).alias("_key")
        )
        .select([
            "entity_id",
            "_key"
        ])
        .unique()
    )

    other_rare = (
        other_tok
        .join(
            other_freq.select([
                "normalized_country",
                "tok"
            ]),
            on=[
                "normalized_country",
                "tok"
            ],
            how="inner"
        )
        .with_columns(
            (
                pl.col("normalized_country")
                + pl.lit(SEP)
                + pl.col("tok")
                + pl.lit(SEP)
                + pl.col("address_num")
            ).alias("_key")
        )
        .select([
            "entity_id",
            "_key"
        ])
        .unique()
    )

    sizes_s1 = (
        s1_rare
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n1")
        )
    )

    sizes_other = (
        other_rare
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n2")
        )
    )

    safe_keys = (
        sizes_s1
        .join(
            sizes_other,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n1")
                * pl.col("_n2")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select("_key")
    )

    joined = (
        s1_rare
        .join(
            safe_keys,
            on="_key",
            how="inner"
        )
        .join(
            other_rare
            .join(
                safe_keys,
                on="_key",
                how="inner"
            ),
            on="_key",
            how="inner",
            suffix="_other"
        )
    )

    return (
        joined
        .select([
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        ])
        .unique()
        .with_columns([
            pl.lit("rare_token_num")
            .alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        ])
    )


# ==========================================================================
# NAME CORE + ADDRESS TOKEN OVERFLOW
# ==========================================================================

def generate_name_core_address_token_overflow_block(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    max_pairs_per_key: int = MAX_PAIRS_PER_BLOCK_KEY,
) -> pl.LazyFrame:

    key_col = "normalized_name_core"

    s1_k = (
        _composite_key(
            s1_lf,
            key_col
        )
        .filter(
            (pl.col(key_col) != "")
            & pl.col(key_col).is_not_null()
        )
    )

    other_k = (
        _composite_key(
            other_lf,
            key_col
        )
        .filter(
            (pl.col(key_col) != "")
            & pl.col(key_col).is_not_null()
        )
    )

    sizes_s1 = (
        s1_k
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n_s1")
        )
    )

    sizes_other = (
        other_k
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n_other")
        )
    )

    estimates = (
        sizes_s1
        .join(
            sizes_other,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n_s1")
                * pl.col("_n_other")
            ).alias("_pair_estimate")
        )
    )

    overflow_keys = (
        estimates
        .filter(
            pl.col("_pair_estimate")
            > max_pairs_per_key
        )
        .select("_key")
    )

    s1_overflow = (
        s1_k
        .join(
            overflow_keys,
            on="_key",
            how="inner"
        )
    )

    other_overflow = (
        other_k
        .join(
            overflow_keys,
            on="_key",
            how="inner"
        )
    )

    s1_addr_tok = (
        s1_overflow
        .select([
            "entity_id",
            "_key",
            "normalized_address"
        ])
        .filter(
            pl.col("normalized_address")
            != ""
        )
        .with_columns(
            pl.col("normalized_address")
            .str.split(" ")
            .alias("_atok")
        )
        .explode("_atok")
        .filter(
            pl.col("_atok")
            .str.len_chars()
            >= MIN_TOKEN_LEN
        )
        .filter(
            ~pl.col("_atok")
            .is_in(list(ADDRESS_STOPWORDS))
        )
        .unique(
            subset=[
                "entity_id",
                "_key",
                "_atok"
            ]
        )
    )

    other_addr_tok = (
        other_overflow
        .select([
            "entity_id",
            "_key",
            "normalized_address"
        ])
        .filter(
            pl.col("normalized_address")
            != ""
        )
        .with_columns(
            pl.col("normalized_address")
            .str.split(" ")
            .alias("_atok")
        )
        .explode("_atok")
        .filter(
            pl.col("_atok")
            .str.len_chars()
            >= MIN_TOKEN_LEN
        )
        .filter(
            ~pl.col("_atok")
            .is_in(list(ADDRESS_STOPWORDS))
        )
        .unique(
            subset=[
                "entity_id",
                "_key",
                "_atok"
            ]
        )
    )

    join_cols = [
        "_key",
        "_atok"
    ]

    sizes_s1_c = (
        s1_addr_tok
        .group_by(join_cols)
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n1")
        )
    )

    sizes_other_c = (
        other_addr_tok
        .group_by(join_cols)
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n2")
        )
    )

    safe_compound = (
        sizes_s1_c
        .join(
            sizes_other_c,
            on=join_cols,
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n1")
                * pl.col("_n2")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select(join_cols)
    )

    s1_addr_tok_f = (
        s1_addr_tok
        .join(
            safe_compound,
            on=join_cols,
            how="inner"
        )
    )

    other_addr_tok_f = (
        other_addr_tok
        .join(
            safe_compound,
            on=join_cols,
            how="inner"
        )
    )

    joined = (
        s1_addr_tok_f
        .select([
            "entity_id"
        ] + join_cols)
        .join(
            other_addr_tok_f
            .select([
                "entity_id"
            ] + join_cols),
            on=join_cols,
            how="inner",
            suffix="_other",
        )
    )

    return (
        joined
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .unique()
        .with_columns(
            pl.lit(
                "name_core_addr_token_overflow"
            ).alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )


# ==========================================================================
# PHONETIC BLOCKING
# ==========================================================================

def _compute_phonetic_lookup(
    tokens_lf: pl.LazyFrame
) -> pl.DataFrame:

    import jellyfish

    distinct_tokens = (
        tokens_lf
        .select("_first_tok")
        .unique()
        .collect()
    )

    tokens_list = (
        distinct_tokens["_first_tok"]
        .to_list()
    )

    codes = [
        jellyfish.metaphone(t)
        if t else ""
        for t in tokens_list
    ]

    return pl.DataFrame({
        "_first_tok": tokens_list,
        "_phon": codes
    })


def generate_phonetic_block_pairs(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    max_pairs_per_key: int = MAX_PAIRS_PER_BLOCK_KEY,
) -> pl.LazyFrame:

    def first_token(col: str) -> pl.Expr:
        return (
            pl.col(col)
            .str.split(" ")
            .list.first()
            .fill_null("")
        )

    s1_base = (
        s1_lf
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core"
        ])
        .with_columns(
            first_token(
                "normalized_name_core"
            ).alias("_first_tok")
        )
        .filter(
            pl.col("_first_tok") != ""
        )
    )

    other_base = (
        other_lf
        .select([
            "entity_id",
            "normalized_country",
            "normalized_name_core"
        ])
        .with_columns(
            first_token(
                "normalized_name_core"
            ).alias("_first_tok")
        )
        .filter(
            pl.col("_first_tok") != ""
        )
    )

    combined_tokens_lf = pl.concat(
        [
            s1_base.select("_first_tok"),
            other_base.select("_first_tok")
        ],
        how="vertical"
    )

    phon_lookup = (
        _compute_phonetic_lookup(
            combined_tokens_lf
        )
        .lazy()
    )

    s1_phon = (
        s1_base
        .join(
            phon_lookup,
            on="_first_tok",
            how="inner"
        )
        .with_columns(
            (
                pl.col("normalized_country")
                .fill_null("")
                + pl.lit(SEP)
                + pl.col("_phon")
            ).alias("_key")
        )
    )

    other_phon = (
        other_base
        .join(
            phon_lookup,
            on="_first_tok",
            how="inner"
        )
        .with_columns(
            (
                pl.col("normalized_country")
                .fill_null("")
                + pl.lit(SEP)
                + pl.col("_phon")
            ).alias("_key")
        )
    )

    sizes_s1 = (
        s1_phon
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n1")
        )
    )

    sizes_other = (
        other_phon
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n2")
        )
    )

    safe_keys = (
        sizes_s1
        .join(
            sizes_other,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n1")
                * pl.col("_n2")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select("_key")
    )

    joined = (
        s1_phon
        .join(
            safe_keys,
            on="_key",
            how="inner"
        )
        .select([
            "entity_id",
            "_key"
        ])
        .join(
            other_phon
            .join(
                safe_keys,
                on="_key",
                how="inner"
            )
            .select([
                "entity_id",
                "_key"
            ]),
            on="_key",
            how="inner",
            suffix="_other",
        )
    )

    return (
        joined
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .unique()
        .with_columns(
            pl.lit(
                "name_phonetic"
            ).alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )


# ==========================================================================
# GENERIC PREFIX RESCUE
# ==========================================================================

def _prefix_key_block(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    text_col: str,
    prefix_len: int,
    max_pairs_per_key: int,
) -> pl.LazyFrame:

    def with_key(lf):

        return (
            lf
            .filter(
                pl.col(text_col)
                .is_not_null()
                & (
                    pl.col(text_col)
                    .str.len_chars()
                    >= prefix_len
                )
            )
            .with_columns(
                (
                    pl.col("normalized_country")
                    .fill_null("")
                    + pl.lit(SEP)
                    + pl.col(text_col)
                    .str.slice(0, prefix_len)
                ).alias("_key")
            )
        )

    s1_p = with_key(s1_lf)
    other_p = with_key(other_lf)

    sizes_s1 = (
        s1_p
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n1")
        )
    )

    sizes_other = (
        other_p
        .group_by("_key")
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n2")
        )
    )

    safe_keys = (
        sizes_s1
        .join(
            sizes_other,
            on="_key",
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n1")
                * pl.col("_n2")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select("_key")
    )

    joined = (
        s1_p
        .join(
            safe_keys,
            on="_key",
            how="inner"
        )
        .select([
            "entity_id",
            "_key"
        ])
        .join(
            other_p
            .join(
                safe_keys,
                on="_key",
                how="inner"
            )
            .select([
                "entity_id",
                "_key"
            ]),
            on="_key",
            how="inner",
            suffix="_other",
        )
    )

    return (
        joined
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .unique()
        .with_columns(
            pl.lit(
                f"prefix_{prefix_len}_{text_col}"
            ).alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )


# ==========================================================================
# CHARACTER N-GRAM BLOCKING
# ==========================================================================

def generate_ngram_block_pairs(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    n: int = 3,
    max_pairs_per_key: int = MAX_PAIRS_PER_BLOCK_KEY,
) -> pl.LazyFrame:

    def build_ngrams(
        lf: pl.LazyFrame
    ) -> pl.LazyFrame:

        base = (
            lf
            .select([
                "entity_id",
                "normalized_country",
                "normalized_name_core"
            ])
            .filter(
                pl.col("normalized_name_core")
                .str.len_chars()
                >= n
            )
            .with_columns(
                pl.col("normalized_name_core")
                .str.replace_all(
                    " ",
                    ""
                )
                .alias("_flat")
            )
        )

        distinct_flat = (
            base
            .select("_flat")
            .unique()
            .collect()
        )

        flat_list = (
            distinct_flat["_flat"]
            .to_list()
        )

        ngram_rows = []

        for s in flat_list:

            grams = (
                {
                    s[i:i + n]
                    for i in range(
                        len(s) - n + 1
                    )
                }
                if len(s) >= n
                else set()
            )

            for g in grams:
                ngram_rows.append(
                    (s, g)
                )

        if not ngram_rows:
            return (
                base
                .with_columns(
                    pl.lit("")
                    .alias("_gram")
                )
                .filter(
                    pl.lit(False)
                )
            )

        ngram_lookup = pl.DataFrame(
            ngram_rows,
            schema=[
                "_flat",
                "_gram"
            ],
            orient="row"
        )

        return base.join(
            ngram_lookup.lazy(),
            on="_flat",
            how="inner"
        )

    s1_ng = build_ngrams(s1_lf)
    other_ng = build_ngrams(other_lf)

    freq_other = (
        other_ng
        .group_by([
            "normalized_country",
            "_gram"
        ])
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_freq_other")
        )
    )

    s1_ng_freq = (
        s1_ng
        .join(
            freq_other,
            on=[
                "normalized_country",
                "_gram"
            ],
            how="inner"
        )
    )

    # IMPORTANT:
    # s1_anchor remains a LazyFrame.
    # Do NOT call .lazy() on it later.
    s1_anchor = (
        s1_ng_freq
        .sort("_freq_other")
        .group_by(
            "entity_id",
            maintain_order=True
        )
        .agg([
            pl.col("normalized_country")
            .first(),

            pl.col("_gram")
            .first(),

            pl.col("_freq_other")
            .first(),
        ])
    )

    key_cols = [
        "normalized_country",
        "_gram"
    ]

    # FIXED: s1_anchor is already LazyFrame.
    anchor_keys = (
        s1_anchor
        .select(key_cols)
        .unique()
    )

    other_restricted = (
        other_ng
        .join(
            anchor_keys,
            on=key_cols,
            how="inner"
        )
    )

    sizes_s1 = (
        s1_anchor
        .group_by(key_cols)
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n1")
        )
    )

    sizes_other = (
        other_restricted
        .group_by(key_cols)
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n2")
        )
    )

    safe_keys = (
        sizes_s1
        .join(
            sizes_other,
            on=key_cols,
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n1")
                * pl.col("_n2")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select(key_cols)
    )

    s1_safe = (
        s1_anchor
        .join(
            safe_keys,
            on=key_cols,
            how="inner"
        )
    )

    other_safe = (
        other_restricted
        .join(
            safe_keys,
            on=key_cols,
            how="inner"
        )
    )

    joined = (
        s1_safe
        .select([
            "entity_id"
        ] + key_cols)
        .join(
            other_safe
            .select([
                "entity_id"
            ] + key_cols),
            on=key_cols,
            how="inner",
            suffix="_other",
        )
    )

    return (
        joined
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .unique()
        .with_columns(
            pl.lit(
                f"ngram_{n}"
            ).alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )


# ==========================================================================
# TRANSLITERATION BLOCKING
# ==========================================================================

def generate_transliteration_block_pairs(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    max_pairs_per_key: int = MAX_PAIRS_PER_BLOCK_KEY,
) -> pl.LazyFrame:

    from unidecode import unidecode

    def build_transliterated(
        lf: pl.LazyFrame
    ) -> pl.LazyFrame:

        base = (
            lf
            .select([
                "entity_id",
                "normalized_country",
                "normalized_name_core"
            ])
            .filter(
                pl.col("normalized_name_core")
                != ""
            )
        )

        distinct_names = (
            base
            .select("normalized_name_core")
            .unique()
            .collect()
        )

        names_list = (
            distinct_names[
                "normalized_name_core"
            ]
            .to_list()
        )

        folded = [
            unidecode(n)
            .strip()
            .lower()
            for n in names_list
        ]

        first_tok = [
            f.split(" ")[0]
            if f
            else ""
            for f in folded
        ]

        lookup = (
            pl.DataFrame({
                "normalized_name_core":
                    names_list,
                "_translit_tok":
                    first_tok
            })
            .filter(
                pl.col("_translit_tok")
                .str.len_chars()
                >= MIN_TOKEN_LEN
            )
        )

        return base.join(
            lookup.lazy(),
            on="normalized_name_core",
            how="inner"
        )

    s1_t = build_transliterated(
        s1_lf
    )

    other_t = build_transliterated(
        other_lf
    )

    key_cols = [
        "normalized_country",
        "_translit_tok"
    ]

    sizes_s1 = (
        s1_t
        .group_by(key_cols)
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n1")
        )
    )

    sizes_other = (
        other_t
        .group_by(key_cols)
        .agg(
            pl.col("entity_id")
            .n_unique()
            .alias("_n2")
        )
    )

    safe_keys = (
        sizes_s1
        .join(
            sizes_other,
            on=key_cols,
            how="inner"
        )
        .with_columns(
            (
                pl.col("_n1")
                * pl.col("_n2")
            ).alias("_pair_estimate")
        )
        .filter(
            pl.col("_pair_estimate")
            <= max_pairs_per_key
        )
        .select(key_cols)
    )

    s1_safe = (
        s1_t
        .join(
            safe_keys,
            on=key_cols,
            how="inner"
        )
    )

    other_safe = (
        other_t
        .join(
            safe_keys,
            on=key_cols,
            how="inner"
        )
    )

    joined = (
        s1_safe
        .select([
            "entity_id"
        ] + key_cols)
        .join(
            other_safe
            .select([
                "entity_id"
            ] + key_cols),
            on=key_cols,
            how="inner",
            suffix="_other",
        )
    )

    return (
        joined
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id"),

            pl.col("entity_id_other")
            .alias("candidate_entity_id"),
        )
        .unique()
        .with_columns(
            pl.lit(
                "transliteration"
            ).alias("block_type"),

            pl.lit(other_label)
            .alias("source"),
        )
    )


# ==========================================================================
# EXISTING PROVEN RESCUE BLOCK REUSE
# ==========================================================================

def reuse_existing_block(
    blocks_dir: Path,
    other_label: str,
    block_name: str,
    written: list[Path],
) -> None:

    path = (
        blocks_dir
        / f"{other_label}_{block_name}.parquet"
    )

    if not path.exists():
        print(
            f"  [warning] Missing proven block: "
            f"{path.name}"
        )
        return

    n = (
        pl.scan_parquet(path)
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )

    print(
        f"  [reuse] {path.name:45} "
        f"{n:>12,} pairs"
    )

    written.append(path)


# ==========================================================================
# GENERATE BLOCK FILES
# ==========================================================================

def generate_and_write_blocks(
    s1_lf: pl.LazyFrame,
    other_lf: pl.LazyFrame,
    other_label: str,
    blocks_dir: Path
) -> list[Path]:

    written = []

    # ------------------------------------------------------
    # EXACT BLOCKS
    # ------------------------------------------------------

    for block_label, key_col in CORE_BLOCK_KEYS.items():

        out_path = (
            blocks_dir
            / f"{other_label}_{block_label}.parquet"
        )

        if out_path.exists():

            n = (
                pl.scan_parquet(out_path)
                .select(pl.len())
                .collect(
                    engine="streaming"
                )
                .item()
            )

            print(
                f"  [reuse] "
                f"{out_path.name:35} "
                f"{n:>12,} pairs"
            )

            written.append(out_path)
            continue

        print(
            f"  [build] "
            f"{other_label}_{block_label}"
        )

        pairs_lf = (
            generate_exact_with_fallback(
                s1_lf,
                other_lf,
                key_col,
                block_label,
                other_label
            )
        )

        pairs_lf.sink_parquet(
            out_path
        )

        n = (
            pl.scan_parquet(out_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"{block_label:15} -> "
            f"{n:>12,} pairs"
        )

        written.append(out_path)

        gc.collect()

    # ------------------------------------------------------
    # TOKEN BLOCKS
    # ------------------------------------------------------

    for (
        block_label,
        text_col,
        stopwords
    ) in [
        (
            "name_token",
            "normalized_name",
            NAME_STOPWORDS
        ),
        (
            "address_token",
            "normalized_address",
            ADDRESS_STOPWORDS
        ),
    ]:

        out_path = (
            blocks_dir
            / f"{other_label}_{block_label}.parquet"
        )

        if out_path.exists():

            n = (
                pl.scan_parquet(out_path)
                .select(pl.len())
                .collect(
                    engine="streaming"
                )
                .item()
            )

            print(
                f"  [reuse] "
                f"{out_path.name:35} "
                f"{n:>12,} pairs"
            )

            written.append(out_path)
            continue

        print(
            f"  [build] "
            f"{other_label}_{block_label}"
        )

        pairs_lf = (
            generate_token_block_pairs(
                s1_lf,
                other_lf,
                text_col,
                stopwords,
                block_label,
                other_label
            )
        )

        pairs_lf.sink_parquet(
            out_path
        )

        n = (
            pl.scan_parquet(out_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"{block_label:15} -> "
            f"{n:>12,} pairs"
        )

        written.append(out_path)

        gc.collect()

    # ------------------------------------------------------
    # EXISTING PREFIX 4
    # ------------------------------------------------------

    prefix_path = (
        blocks_dir
        / f"{other_label}_prefix_4.parquet"
    )

    if prefix_path.exists():

        n = (
            pl.scan_parquet(prefix_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [reuse] "
            f"{prefix_path.name:35} "
            f"{n:>12,} pairs"
        )

        written.append(prefix_path)

    else:

        print(
            f"  [build] "
            f"{other_label}_prefix_4"
        )

        prefix_lf = (
            generate_prefix_block_pairs(
                s1_lf,
                other_lf,
                other_label,
                prefix_len=4,
                max_pairs_per_key=3_000,
            )
        )

        prefix_lf.sink_parquet(
            prefix_path
        )

        n = (
            pl.scan_parquet(prefix_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"prefix_4 -> "
            f"{n:>12,} pairs"
        )

        written.append(prefix_path)

        gc.collect()

    # ------------------------------------------------------
    # RARE TOKEN + ADDRESS NUMBER
    # ------------------------------------------------------

    rare_path = (
        blocks_dir
        / f"{other_label}_rare_token_num.parquet"
    )

    if rare_path.exists():

        n = (
            pl.scan_parquet(rare_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [reuse] "
            f"{rare_path.name:35} "
            f"{n:>12,} pairs"
        )

        written.append(rare_path)

    else:

        print(
            f"  [build] "
            f"{other_label}_rare_token_num"
        )

        rare_lf = (
            generate_rare_token_number_blocks(
                s1_lf,
                other_lf,
                other_label,
                max_token_freq=2500,
                max_pairs_per_key=MAX_PAIRS_PER_BLOCK_KEY,
            )
        )

        rare_lf.sink_parquet(
            rare_path
        )

        n = (
            pl.scan_parquet(rare_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"rare_token_num -> "
            f"{n:>12,} pairs"
        )

        written.append(rare_path)

        gc.collect()

    # ------------------------------------------------------
    # NAME CORE OVERFLOW
    # ------------------------------------------------------

    overflow_path = (
        blocks_dir
        / f"{other_label}_name_core_addr_token_overflow.parquet"
    )

    if overflow_path.exists():

        n = (
            pl.scan_parquet(overflow_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [reuse] "
            f"{overflow_path.name:45} "
            f"{n:>12,} pairs"
        )

        written.append(overflow_path)

    else:

        print(
            f"  [build] "
            f"{other_label}_name_core_addr_token_overflow"
        )

        overflow_lf = (
            generate_name_core_address_token_overflow_block(
                s1_lf,
                other_lf,
                other_label,
                max_pairs_per_key=MAX_PAIRS_PER_BLOCK_KEY,
            )
        )

        overflow_lf.sink_parquet(
            overflow_path
        )

        n = (
            pl.scan_parquet(overflow_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"name_core_addr_token_overflow -> "
            f"{n:>12,} pairs"
        )

        written.append(overflow_path)

        gc.collect()

    # ------------------------------------------------------
    # PHONETIC
    # ------------------------------------------------------

    phon_path = (
        blocks_dir
        / f"{other_label}_name_phonetic.parquet"
    )

    if phon_path.exists():

        n = (
            pl.scan_parquet(phon_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [reuse] "
            f"{phon_path.name:35} "
            f"{n:>12,} pairs"
        )

        written.append(phon_path)

    else:

        print(
            f"  [build] "
            f"{other_label}_name_phonetic"
        )

        phon_lf = (
            generate_phonetic_block_pairs(
                s1_lf,
                other_lf,
                other_label,
                max_pairs_per_key=MAX_PAIRS_PER_BLOCK_KEY,
            )
        )

        phon_lf.sink_parquet(
            phon_path
        )

        n = (
            pl.scan_parquet(phon_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"name_phonetic -> "
            f"{n:>12,} pairs"
        )

        written.append(phon_path)

        gc.collect()

    # ------------------------------------------------------
    # PREFIX RESCUE BLOCKS
    # ------------------------------------------------------

    for col, plen in [
        ("normalized_name_core", 3),
        ("normalized_name_core", 5),
        ("normalized_address", 4),
    ]:

        p_path = (
            blocks_dir
            / f"{other_label}_prefix_{plen}_{col}.parquet"
        )

        if p_path.exists():

            n = (
                pl.scan_parquet(p_path)
                .select(pl.len())
                .collect(
                    engine="streaming"
                )
                .item()
            )

            print(
                f"  [reuse] "
                f"{p_path.name:50} "
                f"{n:>12,} pairs"
            )

            written.append(p_path)

        else:

            print(
                f"  [build] "
                f"{other_label}_prefix_{plen}_{col}"
            )

            p_lf = _prefix_key_block(
                s1_lf,
                other_lf,
                other_label,
                col,
                plen,
                MAX_PAIRS_PER_BLOCK_KEY,
            )

            p_lf.sink_parquet(
                p_path
            )

            n = (
                pl.scan_parquet(p_path)
                .select(pl.len())
                .collect(
                    engine="streaming"
                )
                .item()
            )

            print(
                f"  [{other_label}] "
                f"prefix_{plen}_{col} -> "
                f"{n:>12,} pairs"
            )

            written.append(p_path)

            gc.collect()

    # ------------------------------------------------------
    # PROVEN RESCUE:
    # PREFIX + ADDRESS NUMBER
    #
    # DO NOT REBUILD THESE.
    # These blocks already produced the measured 90.0084%
    # train blocking recall when integrated.
    # ------------------------------------------------------

    print(
        "\n  --- Reusing proven prefix + address-number blocks ---"
    )

    reuse_existing_block(
        blocks_dir,
        other_label,
        "prefix_3_number",
        written,
    )

    reuse_existing_block(
        blocks_dir,
        other_label,
        "prefix_4_number",
        written,
    )

    # ------------------------------------------------------
    # PROVEN RESCUE:
    # ADDRESS TWO TOKEN + NUMBER
    # ------------------------------------------------------

    print(
        "\n  --- Reusing proven address-token-number block ---"
    )

    reuse_existing_block(
        blocks_dir,
        other_label,
        "address_two_token_number",
        written,
    )

    # ------------------------------------------------------
    # N-GRAM
    # ------------------------------------------------------

    ngram_path = (
        blocks_dir
        / f"{other_label}_ngram_3.parquet"
    )

    if ngram_path.exists():

        n = (
            pl.scan_parquet(ngram_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [reuse] "
            f"{ngram_path.name:40} "
            f"{n:>12,} pairs"
        )

        written.append(ngram_path)

    else:

        print(
            f"  [build] "
            f"{other_label}_ngram_3"
        )

        ngram_lf = (
            generate_ngram_block_pairs(
                s1_lf,
                other_lf,
                other_label,
                n=3,
                max_pairs_per_key=MAX_PAIRS_PER_BLOCK_KEY,
            )
        )

        ngram_lf.sink_parquet(
            ngram_path
        )

        n = (
            pl.scan_parquet(ngram_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"ngram_3 -> "
            f"{n:>12,} pairs"
        )

        written.append(ngram_path)

        gc.collect()

    # ------------------------------------------------------
    # TRANSLITERATION
    # ------------------------------------------------------

    translit_path = (
        blocks_dir
        / f"{other_label}_transliteration.parquet"
    )

    if translit_path.exists():

        n = (
            pl.scan_parquet(translit_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [reuse] "
            f"{translit_path.name:40} "
            f"{n:>12,} pairs"
        )

        written.append(translit_path)

    else:

        print(
            f"  [build] "
            f"{other_label}_transliteration"
        )

        translit_lf = (
            generate_transliteration_block_pairs(
                s1_lf,
                other_lf,
                other_label,
                MAX_PAIRS_PER_BLOCK_KEY,
            )
        )

        translit_lf.sink_parquet(
            translit_path
        )

        n = (
            pl.scan_parquet(translit_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        print(
            f"  [{other_label}] "
            f"transliteration -> "
            f"{n:>12,} pairs"
        )

        written.append(translit_path)

        gc.collect()

    return written


# ==========================================================================
# STREAMING DEDUPE & PARTITIONING
# ==========================================================================

def create_partitioned_candidates(
    block_paths: list[Path],
    out_dir: Path,
    num_partitions: int = NUM_CANDIDATE_PARTITIONS,
) -> tuple[list[Path], int]:

    partition_dir = (
        out_dir
        / "candidate_partitions"
    )

    partition_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    part_paths = [
        partition_dir
        / f"candidate_part_{i:03d}.parquet"
        for i in range(num_partitions)
    ]

    if all(
        p.exists()
        for p in part_paths
    ):

        print(
            "\n[reuse] All candidate partitions already exist."
        )

        total_count = sum(
            pl.scan_parquet(p)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
            for p in part_paths
        )

        return (
            part_paths,
            total_count
        )

    print(
        "\n--- Streaming & Deduplicating directly into Partitions ---"
    )

    union_lf = pl.concat(
        [
            pl.scan_parquet(p)
            for p in block_paths
        ],
        how="vertical"
    )

    total_deduped_rows = 0

    for part_id, part_path in enumerate(
        part_paths
    ):

        if part_path.exists():

            cnt = (
                pl.scan_parquet(part_path)
                .select(pl.len())
                .collect(
                    engine="streaming"
                )
                .item()
            )

            total_deduped_rows += cnt

            print(
                f"  [reuse] Partition "
                f"{part_id + 1}/"
                f"{num_partitions}: "
                f"{cnt:,} rows"
            )

            continue

        print(
            f"  [build] Partition "
            f"{part_id + 1}/"
            f"{num_partitions}..."
        )

        part_lf = (
            union_lf
            .filter(
                (
                    pl.col(
                        "source1_entity_id"
                    ).hash()
                    % num_partitions
                )
                == part_id
            )
            .group_by([
                "source1_entity_id",
                "candidate_entity_id",
                "source"
            ])
            .agg(
                pl.col(
                    "block_type"
                )
                .unique()
                .sort()
                .alias("block_types")
            )
            .with_columns(
                pl.col("block_types")
                .list.join(",")
                .alias("block_types")
            )
        )

        part_lf.sink_parquet(
            part_path
        )

        cnt = (
            pl.scan_parquet(part_path)
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        total_deduped_rows += cnt

        print(
            f"         wrote "
            f"{cnt:,} deduped rows -> "
            f"{part_path.name}"
        )

        gc.collect()

    return (
        part_paths,
        total_deduped_rows
    )


# ==========================================================================
# GROUND TRUTH
# ==========================================================================

def prepare_ground_truth(
    gt_path: Path
) -> pl.LazyFrame:

    return (
        pl.scan_csv(
            gt_path,
            separator="\t",
            schema_overrides={
                "source1_entity_id": pl.String,
                "matched_entity_ids": pl.String,
            },
        )
        .filter(
            pl.col("matched_entity_ids")
            .fill_null("")
            .str.strip_chars()
            != ""
        )
        .with_columns(
            pl.col("matched_entity_ids")
            .str.split(",")
            .alias("_ids")
        )
        .explode("_ids")
        .with_columns(
            pl.col("_ids")
            .str.strip_chars()
            .alias("candidate_entity_id")
        )
        .filter(
            pl.col("candidate_entity_id")
            != ""
        )
        .select([
            "source1_entity_id",
            "candidate_entity_id",
        ])
        .unique()
    )


# ==========================================================================
# RECALL EVALUATION
# ==========================================================================

def evaluate_blocking_recall(
    partition_paths: list[Path],
    gt_path: Path
) -> dict:

    print(
        "\n--- Blocking recall evaluation "
        "(partitioned) ---"
    )

    true_pairs_lf = (
        prepare_ground_truth(
            gt_path
        )
    )

    total_true = (
        true_pairs_lf
        .select(pl.len())
        .collect(
            engine="streaming"
        )
        .item()
    )

    print(
        f"[recall] Total true pairs: "
        f"{total_true:,}"
    )

    gt_partitioned = (
        true_pairs_lf
        .with_columns(
            (
                pl.col(
                    "source1_entity_id"
                ).hash()
                % NUM_CANDIDATE_PARTITIONS
            ).alias("_partition")
        )
    )

    total_found = 0
    missed_samples = []

    for partition_id, candidate_path in enumerate(
        partition_paths
    ):

        print(
            f"\n[recall] Partition "
            f"{partition_id + 1}/"
            f"{len(partition_paths)}"
        )

        gt_part = (
            gt_partitioned
            .filter(
                pl.col("_partition")
                == partition_id
            )
            .select([
                "source1_entity_id",
                "candidate_entity_id",
            ])
        )

        candidate_part = (
            pl.scan_parquet(
                candidate_path
            )
            .select([
                "source1_entity_id",
                "candidate_entity_id",
            ])
            .unique()
        )

        found_part = (
            gt_part
            .join(
                candidate_part,
                on=[
                    "source1_entity_id",
                    "candidate_entity_id",
                ],
                how="semi",
            )
        )

        found_count = (
            found_part
            .select(pl.len())
            .collect(
                engine="streaming"
            )
            .item()
        )

        total_found += found_count

        print(
            f"    found: "
            f"{found_count:,}"
        )

        print(
            f"    running total: "
            f"{total_found:,}"
        )

        if len(missed_samples) < 20:

            missed_part = (
                gt_part
                .join(
                    candidate_part,
                    on=[
                        "source1_entity_id",
                        "candidate_entity_id",
                    ],
                    how="anti",
                )
                .limit(
                    20 - len(
                        missed_samples
                    )
                )
                .collect(
                    engine="streaming"
                )
            )

            if missed_part.height:
                missed_samples.extend(
                    missed_part.iter_rows()
                )

        del gt_part
        del candidate_part
        del found_part

        gc.collect()

    missed_count = (
        total_true
        - total_found
    )

    recall = (
        total_found / total_true
        if total_true > 0
        else 1.0
    )

    missed_sample = pl.DataFrame(
        missed_samples,
        schema=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
        orient="row",
    )

    print(
        "\n=========================================="
    )

    print(
        f"Total true pairs : "
        f"{total_true:,}"
    )

    print(
        f"Found true pairs : "
        f"{total_found:,}"
    )

    print(
        f"Missed pairs     : "
        f"{missed_count:,}"
    )

    print(
        f"Blocking recall  : "
        f"{recall:.6%}"
    )

    print(
        "=========================================="
    )

    return {
        "total_true_pairs": total_true,
        "found_pairs": total_found,
        "missed_count": missed_count,
        "recall": recall,
        "missed_sample": missed_sample,
    }


# ==========================================================================
# BUILD CHALLENGE OUTPUT
# ==========================================================================

def build_challenge_output(
    partition_paths: list[Path],
    s1_pq: Path,
    out_path: Path
) -> None:

    print(
        "\n--- Building challenge-format "
        "candidate_pairs.tsv ---"
    )

    temp_dir = (
        out_path.parent
        / "_candidate_output_parts"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    temp_paths = []

    for partition_id, candidate_path in enumerate(
        partition_paths
    ):

        temp_path = (
            temp_dir
            / f"part_{partition_id:03d}.tsv"
        )

        temp_paths.append(
            temp_path
        )

        if temp_path.exists():

            print(
                f"[output] "
                f"{partition_id + 1}/"
                f"{len(partition_paths)} "
                f"[reuse]"
            )

            continue

        print(
            f"[output] "
            f"{partition_id + 1}/"
            f"{len(partition_paths)}"
        )

        part_lf = (
            pl.scan_parquet(
                candidate_path
            )
            .group_by(
                "source1_entity_id"
            )
            .agg(
                pl.col(
                    "candidate_entity_id"
                )
                .unique()
                .sort()
                .alias("_ids")
            )
            .with_columns(
                pl.col("_ids")
                .list.join(",")
                .alias(
                    "candidate_entity_ids"
                )
            )
            .select([
                "source1_entity_id",
                "candidate_entity_ids",
            ])
        )

        part_lf.sink_csv(
            temp_path,
            separator="\t"
        )

        del part_lf

        gc.collect()

    print(
        "\n[output] Adding zero-candidate "
        "S1 entities..."
    )

    all_s1_lf = (
        pl.scan_parquet(
            s1_pq
        )
        .select(
            pl.col("entity_id")
            .alias("source1_entity_id")
        )
        .with_columns(
            (
                pl.col(
                    "source1_entity_id"
                ).hash()
                % NUM_CANDIDATE_PARTITIONS
            ).alias("_partition")
        )
    )

    if out_path.exists():
        out_path.unlink()

    final_part_paths = []

    for partition_id, temp_path in enumerate(
        temp_paths
    ):

        final_part_path = (
            temp_dir
            / f"final_{partition_id:03d}.tsv"
        )

        final_part_paths.append(
            final_part_path
        )

        print(
            f"[output] Final partition "
            f"{partition_id + 1}/"
            f"{len(temp_paths)}"
        )

        candidate_part = (
            pl.scan_csv(
                temp_path,
                separator="\t",
                schema_overrides={
                    "source1_entity_id": pl.String,
                    "candidate_entity_ids": pl.String,
                }
            )
        )

        s1_part = (
            all_s1_lf
            .filter(
                pl.col("_partition")
                == partition_id
            )
            .select(
                "source1_entity_id"
            )
        )

        final_part = (
            s1_part
            .join(
                candidate_part,
                on="source1_entity_id",
                how="left"
            )
            .with_columns(
                pl.col(
                    "candidate_entity_ids"
                )
                .fill_null("")
            )
            .select([
                "source1_entity_id",
                "candidate_entity_ids",
            ])
        )

        final_part.sink_csv(
            final_part_path,
            separator="\t"
        )

        del candidate_part
        del s1_part
        del final_part

        gc.collect()

    print(
        "\n[output] Combining final "
        "partitions..."
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as output_file:

        output_file.write(
            "source1_entity_id"
            "\t"
            "candidate_entity_ids\n"
        )

        for final_part_path in final_part_paths:

            with open(
                final_part_path,
                "r",
                encoding="utf-8"
            ) as input_file:

                next(
                    input_file,
                    None
                )

                for line in input_file:
                    output_file.write(
                        line
                    )

    print(
        f"\n[ok] challenge-format "
        f"candidate_pairs.tsv written to "
        f"{out_path}"
    )


# ==========================================================================
# MAIN PIPELINE
# ==========================================================================

def run(
    train_dir: Path,
    out_dir: Path,
    prefix: str = "train",
    gt_path: Path | None = None,
) -> None:

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    slim_dir = (
        out_dir
        / "slim_parquet"
    )

    blocks_dir = (
        out_dir
        / "blocks"
    )

    blocks_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================================
    # STEP 1 — SLIM PARQUET
    # ======================================================

    print(
        "\n=========================================="
    )

    print(
        f"STEP 1 — Preparing slim Parquet copies ({prefix})"
    )

    print(
        "=========================================="
    )

    # IMPORTANT:
    # Include prefix in filename so train/test cannot
    # accidentally reuse each other's slim parquet.

    s1_pq = ensure_slim_parquet(
        train_dir
        / f"{prefix}_source1.tsv",
        slim_dir
        / f"{prefix}_s1.parquet"
    )

    s2_pq = ensure_slim_parquet(
        train_dir
        / f"{prefix}_source2.tsv",
        slim_dir
        / f"{prefix}_s2.parquet"
    )

    s3_pq = ensure_slim_parquet(
        train_dir
        / f"{prefix}_source3.tsv",
        slim_dir
        / f"{prefix}_s3.parquet"
    )

    s1_lf = pl.scan_parquet(
        s1_pq
    )

    # ======================================================
    # STEP 2 — BLOCKING
    # ======================================================

    print(
        "\n=========================================="
    )

    print(
        "STEP 2 — Blocking"
    )

    print(
        "=========================================="
    )

    print(
        "\n--- S1 x S2 blocking ---"
    )

    s2_lf = pl.scan_parquet(
        s2_pq
    )

    s2_blocks = (
        generate_and_write_blocks(
            s1_lf,
            s2_lf,
            "S2",
            blocks_dir
        )
    )

    del s2_lf

    gc.collect()

    print(
        "\n--- S1 x S3 blocking ---"
    )

    s3_lf = pl.scan_parquet(
        s3_pq
    )

    s3_blocks = (
        generate_and_write_blocks(
            s1_lf,
            s3_lf,
            "S3",
            blocks_dir
        )
    )

    del s3_lf

    gc.collect()

    # ======================================================
    # STEP 3 & 4
    # ======================================================

    print(
        "\n=========================================="
    )

    print(
        "STEP 3 & 4 — Streaming Partitioned Deduplication"
    )

    print(
        "=========================================="
    )

    all_blocks = (
        s2_blocks
        + s3_blocks
    )

    partition_paths, total_candidates = (
        create_partitioned_candidates(
            all_blocks,
            out_dir,
            num_partitions=NUM_CANDIDATE_PARTITIONS,
        )
    )

    print(
        f"\nTotal deduped candidate pairs across partitions: "
        f"{total_candidates:,}"
    )

    # ======================================================
    # STEP 5 — RECALL
    # ======================================================

    if prefix == "train":

        resolved_gt_path = gt_path

        if resolved_gt_path is None:

            candidate_locations = [
                train_dir.parent
                / "train"
                / "train_ground_truth.tsv",

                train_dir
                / "train_ground_truth.tsv",
            ]

            resolved_gt_path = next(
                (
                    p
                    for p in candidate_locations
                    if p.exists()
                ),
                None
            )

        if (
            resolved_gt_path is None
            or not resolved_gt_path.exists()
        ):

            raise FileNotFoundError(
                "Training ground truth not found."
            )

        result = (
            evaluate_blocking_recall(
                partition_paths,
                resolved_gt_path
            )
        )

        print(
            "\nTrue pairs in ground truth  : "
            f"{result['total_true_pairs']:,}"
        )

        print(
            "True pairs found by blocking: "
            f"{result['found_pairs']:,}"
        )

        print(
            "Missed true pairs           : "
            f"{result['missed_count']:,}"
        )

        print(
            "Blocking recall             : "
            f"{result['recall']:.4%}"
        )

    else:

        print(
            "\n[skip] prefix='test' has no ground truth."
        )

    # ======================================================
    # STEP 6 — REDUCTION
    # ======================================================

    n_s1 = (
        pl.scan_parquet(s1_pq)
        .select(pl.len())
        .collect(
            engine="streaming"
        )
        .item()
    )

    n_s2 = (
        pl.scan_parquet(s2_pq)
        .select(pl.len())
        .collect(
            engine="streaming"
        )
        .item()
    )

    n_s3 = (
        pl.scan_parquet(s3_pq)
        .select(pl.len())
        .collect(
            engine="streaming"
        )
        .item()
    )

    brute_force = (
        n_s1 * n_s2
    ) + (
        n_s1 * n_s3
    )

    print(
        "\n=========================================="
    )

    print(
        "Candidate reduction"
    )

    print(
        "=========================================="
    )

    print(
        f"Brute-force pair count: "
        f"{brute_force:,}"
    )

    print(
        f"Actual candidate count: "
        f"{total_candidates:,}"
    )

    print(
        f"Reduction ratio: "
        f"{1 - total_candidates / brute_force:.6%}"
    )

    # ======================================================
    # STEP 7 — OUTPUT
    # ======================================================

    challenge_out = (
        out_dir
        / "candidate_pairs.tsv"
    )

    build_challenge_output(
        partition_paths,
        s1_pq,
        challenge_out
    )

    print(
        "\n=========================================="
    )

    print(
        "BLOCKING STAGE COMPLETE"
    )

    print(
        "=========================================="
    )


# ==========================================================================
# CLI
# ==========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-dir",
        default="dataset/normalized",
        help="Directory containing normalized TSV files.",
    )

    parser.add_argument(
        "--out-dir",
        default="dataset/candidates",
    )

    parser.add_argument(
        "--prefix",
        default="train",
        choices=[
            "train",
            "test"
        ],
    )

    parser.add_argument(
        "--gt-path",
        default=None,
        help="Path to train_ground_truth.tsv.",
    )

    args = parser.parse_args()

    run(
        Path(args.train_dir),
        Path(args.out_dir),
        prefix=args.prefix,
        gt_path=(
            Path(args.gt_path)
            if args.gt_path
            else None
        ),
    )
