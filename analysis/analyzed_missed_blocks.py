import argparse
from pathlib import Path
import polars as pl


def main(blocks_dir, missed_path, output):

    print("Loading current missed true pairs...")

    missed = pl.read_parquet(missed_path).select([
        "source1_entity_id",
        "candidate_entity_id"
    ]).unique()

    print(f"Missed true pairs: {missed.height:,}")

    block_files = sorted(
        p for p in blocks_dir.glob("*.parquet")
        if "_TEST" not in p.name
    )

    print(f"Production blocks: {len(block_files)}")

    results = []

    for i, path in enumerate(block_files, 1):

        hits = (
            pl.scan_parquet(str(path))
            .select([
                "source1_entity_id",
                "candidate_entity_id"
            ])
            .join(
                missed.lazy(),
                on=[
                    "source1_entity_id",
                    "candidate_entity_id"
                ],
                how="semi"
            )
            .unique()
            .collect(engine="streaming")
        )

        n = hits.height

        results.append({
            "block_file": path.name,
            "missed_pairs_recovered": n,
            "missed_recall_pct": 100.0 * n / missed.height
        })

        print(
            f"{i:02d}/{len(block_files)} "
            f"{path.name:<55} "
            f"{n:>8,} "
            f"({100.0 * n / missed.height:.4f}%)"
        )

    result_df = (
        pl.DataFrame(results)
        .sort(
            "missed_pairs_recovered",
            descending=True
        )
    )

    result_df.write_csv(
        output,
        separator="\t"
    )

    print()
    print("Saved:", output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--blocks-dir", required=True)
    parser.add_argument("--missed-path", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(
        Path(args.blocks_dir),
        Path(args.missed_path),
        Path(args.output)
    )
