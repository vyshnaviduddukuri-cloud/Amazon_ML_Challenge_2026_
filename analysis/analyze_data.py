

#import pandas as pd
#
#train_s1 = pd.read_csv(
#    "dataset/train/train_source1.tsv",
#    sep="\t"
#)
#
#train_s2 = pd.read_csv(
#    "dataset/train/train_source2.tsv",
#    sep="\t"
#)
#
#train_s3 = pd.read_csv(
#    "dataset/train/train_source3.tsv",
#    sep="\t"
#)
#
#ground_truth = pd.read_csv(
#    "dataset/train/train_ground_truth.tsv",
#    sep="\t"
#)
#
#print("Source 1 shape:", train_s1.shape)
#print("Source 2 shape:", train_s2.shape)
#print("Source 3 shape:", train_s3.shape)
#print("Ground truth shape:", ground_truth.shape)
#
#print("\nSource 1 columns:")
#print(train_s1.columns.tolist())
#
#print("\nSource 2 columns:")
#print(train_s2.columns.tolist())
#
#print("\nSource 3 columns:")
#print(train_s3.columns.tolist())
#
#print("\nGround truth columns:")
#print(ground_truth.columns.tolist())
#
#print("\n===== SOURCE 1 SAMPLE =====")
#print(train_s1.head(10).to_string(index=False))
#
#print("\n===== SOURCE 2 SAMPLE =====")
#print(train_s2.head(10).to_string(index=False))
#
#print("\n===== SOURCE 3 SAMPLE =====")
#print(train_s3.head(10).to_string(index=False))
#
#print("\n===== GROUND TRUTH SAMPLE =====")
#print(ground_truth.head(10).to_string(index=False))
#
#print("\n===== MISSING VALUES =====")
#
#print("\nSource 1:")
#print(train_s1.isna().sum())
#
#print("\nSource 2:")
#print(train_s2.isna().sum())
#
#print("\nSource 3:")
#print(train_s3.isna().sum())
#
#print("\n===== COUNTRIES =====")
#
#print("\nSource 1:")
#print(train_s1["country"].value_counts(dropna=False))
#
#print("\nSource 2:")
#print(train_s2["country"].value_counts(dropna=False))
#
#print("\nSource 3:")
#print(train_s3["country"].value_counts(dropna=False))
#
#print("\n===== DUPLICATE NAMES =====")
#
#print(
#    "Source 1 duplicate names:",
#    train_s1["business_name"].duplicated().sum()
#)
#
#print(
#    "Source 2 duplicate names:",
#    train_s2["business_name"].duplicated().sum()
#)
#
#print(
#    "Source 3 duplicate names:",
#    train_s3["business_name"].duplicated().sum()
#)
#
#print("\n===== DUPLICATE ADDRESSES =====")
#
#print(
#    "Source 1 duplicate addresses:",
#    train_s1["business_address"].duplicated().sum()
#)
#
#print(
#    "Source 2 duplicate addresses:",
#    train_s2["business_address"].duplicated().sum()
#)
#
#print(
#    "Source 3 duplicate addresses:",
#    train_s3["business_address"].duplicated().sum()
#)
#
#print("\n===== GROUND TRUTH EXAMPLES =====")
#
#for _, row in ground_truth.head(20).iterrows():
#    print(
#        row["source1_entity_id"],
#        "→",
#        row["matched_entity_ids"]
#    )
#
#    ground_truth["match_count"] = (
#    ground_truth["matched_entity_ids"]
#    .fillna("")
#    .apply(
#        lambda x: 0 if x == "" else len(x.split(","))
#    )
#)
#
#print("\n===== MATCH COUNT =====")
#
#print(
#    ground_truth["match_count"].value_counts().sort_index()
#)
#
#print(
#    "\nMaximum matches for one Source 1 entity:",
#    ground_truth["match_count"].max()
#)
#
#missing_gt = set(train_s1["entity_id"]) - set(
#    ground_truth["source1_entity_id"]
#)
#
#print(
#    "\nSource 1 entities missing from ground truth:",
#    len(missing_gt)
#)
#
#print("\n===== ID PREFIX CHECK =====")
#
#print(train_s1["entity_id"].str[:3].value_counts())
#print(train_s2["entity_id"].str[:3].value_counts())
#print(train_s3["entity_id"].str[:3].value_counts())
#
#

import pandas as pd

test_s1 = pd.read_csv(
    "dataset/test/test_source1.tsv",
    sep="\t"
)

test_s2 = pd.read_csv(
    "dataset/test/test_source2.tsv",
    sep="\t"
)

test_s3 = pd.read_csv(
    "dataset/test/test_source3.tsv",
    sep="\t"
)

print("===== SHAPES =====")

print("Test Source 1 shape:", test_s1.shape)
print("Test Source 2 shape:", test_s2.shape)
print("Test Source 3 shape:", test_s3.shape)


print("\n===== COLUMNS =====")

print("Source 1:", test_s1.columns.tolist())
print("Source 2:", test_s2.columns.tolist())
print("Source 3:", test_s3.columns.tolist())


print("\n===== SOURCE 1 SAMPLE =====")
print(test_s1.head(10).to_string(index=False))

print("\n===== SOURCE 2 SAMPLE =====")
print(test_s2.head(10).to_string(index=False))

print("\n===== SOURCE 3 SAMPLE =====")
print(test_s3.head(10).to_string(index=False))


print("\n===== MISSING VALUES =====")

print("\nTest Source 1:")
print(test_s1.isna().sum())

print("\nTest Source 2:")
print(test_s2.isna().sum())

print("\nTest Source 3:")
print(test_s3.isna().sum())


print("\n===== COUNTRIES =====")

print("\nTest Source 1:")
print(test_s1["country"].value_counts(dropna=False))

print("\nTest Source 2:")
print(test_s2["country"].value_counts(dropna=False))

print("\nTest Source 3:")
print(test_s3["country"].value_counts(dropna=False))


print("\n===== DUPLICATE NAMES =====")

print(
    "Test Source 1 duplicate names:",
    test_s1["business_name"].duplicated().sum()
)

print(
    "Test Source 2 duplicate names:",
    test_s2["business_name"].duplicated().sum()
)

print(
    "Test Source 3 duplicate names:",
    test_s3["business_name"].duplicated().sum()
)


print("\n===== DUPLICATE ADDRESSES =====")

print(
    "Test Source 1 duplicate addresses:",
    test_s1["business_address"].duplicated().sum()
)

print(
    "Test Source 2 duplicate addresses:",
    test_s2["business_address"].duplicated().sum()
)

print(
    "Test Source 3 duplicate addresses:",
    test_s3["business_address"].duplicated().sum()
)


print("\n===== ID PREFIX CHECK =====")

print("Test Source 1:")
print(test_s1["entity_id"].str[:3].value_counts())

print("\nTest Source 2:")
print(test_s2["entity_id"].str[:3].value_counts())

print("\nTest Source 3:")
print(test_s3["entity_id"].str[:3].value_counts())


print("\n===== FRANCE CHECK =====")

print("Source 1 France:")
print((test_s1["country"] == "France").sum())

print("Source 2 France:")
print((test_s2["country"] == "France").sum())

print("Source 3 France:")
print((test_s3["country"] == "France").sum())


print("\n===== UNIQUE COUNTRIES =====")

all_test_countries = sorted(
    set(test_s1["country"].dropna())
    | set(test_s2["country"].dropna())
    | set(test_s3["country"].dropna())
)

print(all_test_countries)
