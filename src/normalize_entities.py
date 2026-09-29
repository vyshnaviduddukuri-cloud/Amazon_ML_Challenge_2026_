import argparse
import re
import unicodedata
from pathlib import Path

import polars as pl

LEGAL_SUFFIXES = {
    "US": [
        "incorporated", "inc", "corporation", "corp", "company", "co",
        "llc", "l l c", "llp", "l l p", "ltd", "limited", "lp",
        "pllc", "pc",
    ],
    "INDIA": [
        "private limited", "pvt ltd", "pvt. ltd.", "pvt", "private",
        "limited", "ltd", "llp", "l l p",
        "proprietorship", "enterprises", "enterprise",
    ],
    "FRANCE": [
        "societe anonyme", "societe a responsabilite limitee",
        "sarl", "sas", "sasu", "sa", "eurl", "sci", "ets",
        "etablissements", "cie", "compagnie",
    ],
}

GENERIC_SUFFIXES = ["group", "holdings", "international", "intl"]

US_ADDRESS_ABBREV = {
    r"\bstreet\b": "st", r"\broad\b": "rd", r"\bavenue\b": "ave",
    r"\bboulevard\b": "blvd", r"\bdrive\b": "dr", r"\blane\b": "ln",
    r"\bsuite\b": "ste", r"\bapartment\b": "apt", r"\bfloor\b": "fl",
    r"\bnorth\b": "n", r"\bsouth\b": "s", r"\beast\b": "e", r"\bwest\b": "w",
}

INDIA_ADDRESS_ABBREV = {
    r"\broad\b": "rd", r"\bnagar\b": "ngr", r"\bcolony\b": "cln",
    r"\bnear\b": "nr", r"\bopposite\b": "opp", r"\bfloor\b": "fl",
    r"\bcross\b": "x", r"\bmain\b": "mn",
}

FRANCE_ADDRESS_ABBREV = {
    r"\brue\b": "r", r"\bavenue\b": "av", r"\bboulevard\b": "bd",
    r"\bplace\b": "pl", r"\bsaint\b": "st", r"\bsainte\b": "ste",
    r"\bcedex\b": "",
}

ADDRESS_ABBREV_BY_COUNTRY = {
    "US": US_ADDRESS_ABBREV,
    "INDIA": INDIA_ADDRESS_ABBREV,
    "FRANCE": FRANCE_ADDRESS_ABBREV,
}


COUNTRY_CANONICAL = {
    "us": "US", "usa": "US", "united states": "US",
    "united states of america": "US",
    "india": "INDIA", "in": "INDIA", "bharat": "INDIA",
    "france": "FRANCE", "fr": "FRANCE",
}


def normalize_country(raw) -> str:
    """Map equivalent country strings (USA / United States, India / IN, ...)
    to a canonical value. Unknown values pass through uppercased rather
    than being dropped, so the pipeline degrades gracefully on new
    countries instead of breaking the country-agnostic promise."""
    if raw is None:
        return ""
    key = unicodedata.normalize("NFKC", str(raw)).strip().lower()
    return COUNTRY_CANONICAL.get(key, key.upper())



_PUNCT_RE = re.compile(r"[^\w\s&]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _base_clean(text) -> str:
    """Unicode normalize -> remove accents/diacritics -> casefold
    (script-aware, preserves non-Latin scripts) -> standardize &
    -> strip punctuation -> collapse whitespace. This never deletes
    or overwrites the raw value; it only ever produces a NEW derived string.
    """
    if text is None:
        return ""

    t = unicodedata.normalize("NFKD", str(text))
    t = "".join(
        ch for ch in t
        if not unicodedata.combining(ch)
    )


    t = t.casefold()
    t = t.replace("&", " and ")
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()

    return t

def _strip_trailing_suffix(words: list, suffixes: list) -> list:
    """Remove a recognized legal-suffix phrase from the END of a
    tokenized name only. Never touches suffix-like words in the middle
    of a name, since those are often part of the real business name."""
    changed = True
    while changed and words:
        changed = False
        for suf in sorted(suffixes, key=len, reverse=True):
            suf_words = suf.split()
            n = len(suf_words)
            if n and words[-n:] == suf_words:
                words = words[:-n]
                changed = True
                break
    return words


def normalize_name(raw_name, country) -> str:
    """Cleans case/punctuation/&, but keeps legal suffixes. This is the
    'normalized_name' column — a light-touch cleanup, not a rewrite."""
    return _base_clean(raw_name)


def normalize_name_core(raw_name, country) -> str:
    """A SEPARATE, more aggressive representation: normalized_name with
    a recognized trailing legal suffix removed. Kept alongside (not
    instead of) normalized_name because suffix-stripping is occasionally
    wrong and matching features may want both views."""
    t = _base_clean(raw_name)
    if not t:
        return t
    country_key = normalize_country(country)
    suffixes = LEGAL_SUFFIXES.get(country_key, []) + GENERIC_SUFFIXES
    words = _strip_trailing_suffix(t.split(), suffixes)
    cleaned = " ".join(words).strip()
    # Guard against stripping everything away.
    return cleaned if cleaned else t


def normalize_address(raw_address, country) -> str:
    """Missing addresses stay as '' (a valid, present representation) —
    they are never dropped, and the row is never removed."""
    if raw_address is None or str(raw_address).strip() == "":
        return ""
    t = _base_clean(raw_address)
    country_key = normalize_country(country)
    abbrev_map = ADDRESS_ABBREV_BY_COUNTRY.get(country_key, {})
    for pattern, repl in abbrev_map.items():
        t = re.sub(pattern, repl, t)
    return _WS_RE.sub(" ", t).strip()

def build_normalized_lazyframe(path: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(path, separator="\t", infer_schema_length=10_000)

    lf = lf.with_columns(
        pl.col("country")
        .map_elements(normalize_country, return_dtype=pl.String)
        .alias("normalized_country")
    )

    lf = lf.with_columns(
        pl.struct(["business_name", "normalized_country"])
        .map_elements(
            lambda row: normalize_name(row["business_name"], row["normalized_country"]),
            return_dtype=pl.String,
        )
        .alias("normalized_name"),
        pl.struct(["business_address", "normalized_country"])
        .map_elements(
            lambda row: normalize_address(row["business_address"], row["normalized_country"]),
            return_dtype=pl.String,
        )
        .alias("normalized_address"),
    )

    lf = lf.with_columns(
        pl.struct(["normalized_name", "normalized_country"])
        .map_elements(
            lambda row: normalize_name_core(row["normalized_name"], row["normalized_country"]),
            return_dtype=pl.String,
        )
        .alias("normalized_name_core")
    )

    return lf


def normalize_file(in_path: Path, out_path: Path) -> None:
    lf = build_normalized_lazyframe(in_path)
    lf.sink_csv(out_path, separator="\t")
    print(f"[ok] {in_path.name} -> {out_path.name}  "
          f"(original columns untouched; added normalized_name, "
          f"normalized_name_core, normalized_address, normalized_country)")

def validate_on_matched_sample(train_dir: Path, n_samples: int = 100, seed: int = 42) -> None:
    s1 = pl.read_csv(train_dir / "train_source1.tsv", separator="\t")
    s2 = pl.read_csv(train_dir / "train_source2.tsv", separator="\t")
    s3 = pl.read_csv(train_dir / "train_source3.tsv", separator="\t")
    gt = pl.read_csv(train_dir / "train_ground_truth.tsv", separator="\t")

    s1_lookup = {r["entity_id"]: r for r in s1.to_dicts()}
    s2_lookup = {r["entity_id"]: r for r in s2.to_dicts()}
    s3_lookup = {r["entity_id"]: r for r in s3.to_dicts()}

    matched_gt = gt.filter(
        pl.col("matched_entity_ids").fill_null("").str.strip_chars() != ""
    ).sample(n=n_samples, seed=seed)

    name_conv = name_core_conv = addr_conv = country_conv = total_pairs = 0

    for row in matched_gt.to_dicts():
        s1_row = s1_lookup.get(row["source1_entity_id"])
        if s1_row is None:
            continue

        s1_country = normalize_country(s1_row["country"])
        s1_name = normalize_name(s1_row["business_name"], s1_country)
        s1_name_core = normalize_name_core(s1_row["business_name"], s1_country)
        s1_addr = normalize_address(s1_row["business_address"], s1_country)

        matched_ids = [x.strip() for x in row["matched_entity_ids"].split(",") if x.strip()]
        for mid in matched_ids:
            lookup = s2_lookup if mid.startswith("S2-") else s3_lookup if mid.startswith("S3-") else None
            if lookup is None:
                continue
            m_row = lookup.get(mid)
            if m_row is None:
                continue

            m_country = normalize_country(m_row["country"])
            m_name = normalize_name(m_row["business_name"], m_country)
            m_name_core = normalize_name_core(m_row["business_name"], m_country)
            m_addr = normalize_address(m_row["business_address"], m_country)

            total_pairs += 1
            name_match = s1_name == m_name
            core_match = s1_name_core == m_name_core
            addr_match = s1_addr == m_addr and s1_addr != ""
            country_match = s1_country == m_country

            name_conv += name_match
            name_core_conv += core_match
            addr_conv += addr_match
            country_conv += country_match

            flag = "NAME_OK  " if name_match else ("CORE_OK  " if core_match else "DIFF     ")
            print(f"{flag}| S1: {s1_row['business_name']!r:40} -> {s1_name!r:30} | core: {s1_name_core!r:25}")
            print(f"         | {mid}: {m_row['business_name']!r:35} -> {m_name!r:30} | core: {m_name_core!r:25}")
            print(f"         | addr S1: {s1_addr!r:35} addr M: {m_addr!r:35} | country: {s1_country} vs {m_country}\n")

    print("=" * 100)
    print(f"Pairs inspected           : {total_pairs}")
    print(f"normalized_name matches   : {name_conv}/{total_pairs}")
    print(f"normalized_name_core match: {name_core_conv}/{total_pairs}  "
          f"(should be >= normalized_name matches)")
    print(f"normalized_address matches (non-empty): {addr_conv}/{total_pairs}")
    print(f"normalized_country matches: {country_conv}/{total_pairs}")
    print("\nInspect the DIFF/CORE_OK rows above for the next rules to add — "
          "typos, transliteration, and word-order changes are matching-stage "
          "problems, not normalization-stage ones; don't try to force them "
          "to converge here.")

def run_unit_tests() -> None:
    cases = [
        ("ABC & Co., Pvt. Ltd.", "India", "abc and co pvt ltd", "abc and co"),
        ("Acme Inc.", "US", "acme inc", "acme"),
        ("Acme  INC", "US", "acme inc", "acme"),  # double space + case
        ("Café Résumé SARL", "France", "cafe resume sarl", "cafe resume"),
        ("Limited Editions Cafe", "US", "limited editions cafe", "limited editions cafe"),  # suffix-in-middle guard
        ("Sons & Daughters LLC", "US", "sons and daughters llc", "sons and daughters"),
        ("", "US", "", ""),
    ]
    failures = 0
    for raw, country, exp_name, exp_core in cases:
        country_key = normalize_country(country)
        got_name = normalize_name(raw, country_key)
        got_core = normalize_name_core(raw, country_key)
        ok_name = got_name == exp_name
        ok_core = got_core == exp_core
        status = "PASS" if (ok_name and ok_core) else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] raw={raw!r}")
        print(f"       normalized_name:      got={got_name!r} expected={exp_name!r}")
        print(f"       normalized_name_core: got={got_core!r} expected={exp_core!r}")

    addr_cases = [
        ("123 Main Street, Suite 4", "US", None),   # exercise, no fixed expected
        (None, "US", ""),
        ("", "India", ""),
        ("12 Rue de Paris", "France", None),
    ]
    print("\n-- address normalization spot checks (inspect manually) --")
    for raw, country, _ in addr_cases:
        country_key = normalize_country(country)
        print(f"raw={raw!r:35} country={country_key:8} -> {normalize_address(raw, country_key)!r}")

    country_cases = [
        ("USA", "US"), ("United States", "US"), ("us", "US"),
        ("India", "INDIA"), ("IN", "INDIA"), ("Bharat", "INDIA"),
        ("France", "FRANCE"), ("fr", "FRANCE"),
        ("Germany", "GERMANY"),  # unknown -> passthrough uppercased, not dropped
    ]
    print("\n-- country normalization checks --")
    for raw, expected in country_cases:
        got = normalize_country(raw)
        status = "PASS" if got == expected else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"[{status}] {raw!r:20} -> {got!r:10} expected {expected!r}")

    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--out-dir", default="dataset/normalized")
    parser.add_argument("--validate-sample", action="store_true",
                         help="Step 2D: run convergence check on 100 sampled true matches")
    parser.add_argument("--unit-test", action="store_true",
                         help="Step 2E: run unit tests on individual normalization rules")
    parser.add_argument("--run-all", action="store_true",
                         help="Step 2F/2G: normalize every train/test file "
                              "(DO NOT run until 2D/2E pass and rules are finalized)")
    parser.add_argument("--i-confirm-2d-2e-are-done", action="store_true",
                         help="Required alongside --run-all as a deliberate confirmation gate")
    args = parser.parse_args()

    if args.unit_test:
        run_unit_tests()

    if args.validate_sample:
        validate_on_matched_sample(Path(args.train_dir))

    if args.run_all:
        if not args.i_confirm_2d_2e_are_done:
            print("Refusing to run full normalization: pass "
                  "--i-confirm-2d-2e-are-done once --validate-sample and "
                  "--unit-test output has actually been reviewed. "
                  "We are still at Step 2A/2D per the agreed checklist.")
        else:
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
                normalize_file(Path(args.train_dir) / fname, out_dir / fname)
            for fname in ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]:
                normalize_file(Path(args.test_dir) / fname, out_dir / fname)

    if not (args.validate_sample or args.unit_test or args.run_all):
        parser.print_help()
