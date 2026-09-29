"""
Per-block contribution audit — Business Entity Resolution Challenge

Why this matters NOW specifically: the challenge brief confirms
candidate_pairs.tsv counts toward final ranking, and "the approach that
generates a smaller candidate set per Source 1 entity will be ranked
higher... beyond the public/private leaderboard." That means every
block you keep needs to earn its candidate-volume cost, not just its
recall contribution in isolation.

This measures, for EACH existing block file independently:
    - how many candidate pairs it contributes
    - how many TRUE pairs (from ground truth) it finds ON ITS OWN
    - recall-per-million-candidates for that block alone

IMPORTANT CAVEAT — read before acting on this: this is SOLO recall,
not MARGINAL/UNIQUE recall. A block can look "inefficient" here while
still being fully redundant with a cheaper block (in which case cutting
it costs nothing), or it can look "efficient" here while all its true
pairs are ALSO found by another block (in which case cutting it still
costs nothing). This script is a fast first screen to find obviously
low-value blocks (high candidate cost, low solo recall — cut these
without further analysis) and obviously high-value blocks (worth
keeping regardless of overlap). For any block that's ambiguous under
this metric, a true leave-one-out marginal-recall check is the correct
next step, at higher compute cost — this script tells you which
block(s), if any, are worth paying for that deeper check.

Run:
    python audit_block_contribution.py \
        --blocks-dir dataset/candidates/blocks \
        --gt-path dataset/train/train_ground_truth.tsv \
        --out-dir dataset/block_audit
"""

import argparse
from pathlib import Path

import polars as pl


def prepare_true_pairs(gt_path: Path) -> pl.LazyFrame:
    return (
        pl.scan_csv(gt_path, separator="\t")
        .filter(pl.col("matched_entity_ids").fill_null("").str.strip_chars() != "")
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("_ids"))
        .explode("_ids")
        .with_columns(pl.col("_ids").str.strip_chars().alias("candidate_entity_id"))
        .filter(pl.col("candidate_entity_id") != "")
        .select(["source1_entity_id", "candidate_entity_id"])
        .unique()
    )


def run(blocks_dir: Path, gt_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    true_pairs_lf = prepare_true_pairs(gt_path)
    total_true = true_pairs_lf.select(pl.len()).collect(engine="streaming").item()
    print(f"Total true pairs in ground truth: {total_true:,}\n")

    block_files = sorted(
    p for p in blocks_dir.glob("*.parquet")
    if "_TEST" not in p.name
)
    if not block_files:
        print(f"[warn] No .parquet files found in {blocks_dir}")
        return

    rows = []
    for block_path in block_files:
        block_lf = (
            pl.scan_parquet(block_path)
            .select(["source1_entity_id", "candidate_entity_id"])
            .unique()
        )
        n_candidates = block_lf.select(pl.len()).collect(engine="streaming").item()

        found_lf = true_pairs_lf.join(
            block_lf, on=["source1_entity_id", "candidate_entity_id"], how="semi"
        )
        n_true_found = found_lf.select(pl.len()).collect(engine="streaming").item()

        recall_solo = n_true_found / total_true if total_true else float("nan")
        recall_per_million = (n_true_found / n_candidates * 1_000_000) if n_candidates else float("nan")

        rows.append({
            "block_file": block_path.name,
            "n_candidates": n_candidates,
            "n_true_pairs_found_solo": n_true_found,
            "solo_recall_pct": recall_solo * 100,
            "recall_per_million_candidates": recall_per_million,
        })
        print(f"  {block_path.name:45} candidates={n_candidates:>12,}  "
              f"solo_true_found={n_true_found:>10,}  "
              f"solo_recall={recall_solo:.4%}  "
              f"recall/1M={recall_per_million:,.2f}")

    audit = pl.DataFrame(rows).sort("recall_per_million_candidates", descending=True)
    audit_path = out_dir / "block_contribution_audit.tsv"
    audit.write_csv(audit_path, separator="\t")

    print(f"\n[ok] full audit table -> {audit_path}\n")

    print("=== Ranked by recall-per-million-candidates (best first) ===")
    print(audit)

    total_candidates_all_blocks = audit["n_candidates"].sum()
    print(f"\nSum of candidates across all block FILES (before global dedup): {total_candidates_all_blocks:,}")
    print("(Your actual deduped candidate_pairs.tsv total will be smaller than this sum, "
          "since blocks overlap — but this sum is still useful for spotting which single "
          "block is disproportionately large.)")

    print("\n=== Candidates for pruning (low recall/candidate ratio) ===")
    median_ratio = audit["recall_per_million_candidates"].median()
    low_value = audit.filter(pl.col("recall_per_million_candidates") < median_ratio / 4)
    if low_value.height:
        print(low_value)
        print("\nThese contribute far below the median recall-per-candidate ratio.")
        print("Before cutting any of them: check whether their solo-found true pairs are")
        print("ALSO found by a higher-ranked block (a proper marginal/leave-one-out check).")
        print("If yes, cutting is free — smaller candidate_pairs.tsv, same final recall.")
    else:
        print("No block falls dramatically below the median — no obvious prune candidates")
        print("from this first-pass screen. A full leave-one-out analysis would be needed")
        print("to find more subtle redundancy.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks-dir", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--out-dir", default="dataset/block_audit")
    args = parser.parse_args()
    run(Path(args.blocks_dir), Path(args.gt_path), Path(args.out_dir))
