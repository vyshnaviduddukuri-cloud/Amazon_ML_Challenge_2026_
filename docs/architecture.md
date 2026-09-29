ML Challenge 2026: Business Entity Resolution Solution

Team Name: [algorythm]
Team Members: [G.TanuSree D.Vyshnavi G.Rishika K.Sri Gowri]
Submission Date: 25 - 27 September 2026

1. Executive Summary

We developed a large-scale hybrid entity-resolution pipeline to match business records across multiple data sources. The solution combines data normalization, multi-strategy blocking, fuzzy string similarity, feature engineering, and a LightGBM classifier to efficiently identify likely entity matches.

The blocking pipeline reduced approximately 22.77 trillion brute-force comparisons to 347.6 million candidate pairs, achieving 90.81% blocking recall and a 99.99847% reduction in the comparison space. The final submitted solution achieved an F₀.₅ score of 0.784 and ranked 4,521 out of 10,650 teams.

2. Methodology
2.1 Problem Analysis

The datasets contained millions of business records with variations in business names, addresses, and formatting across different sources.

During exploratory data analysis, we identified several important characteristics:

🏢 Business names frequently contained variations in capitalization, punctuation, abbreviations, and legal suffixes.
📍 Addresses contained significant formatting differences and abbreviations.
🌎 Records belonged to multiple countries, making country-aware normalization useful.
🔤 Some entities had name variations that could not be resolved through exact matching alone.
🔁 Duplicate names and addresses were common, making naive exact matching insufficient.
❌ Some records contained missing names or addresses.
🌐 Transliteration was useful for handling certain cross-script/name variations.
📊 The number of true matches per Source 1 entity varied considerably, including entities with multiple corresponding records.

Because the datasets contained millions of records, a brute-force comparison approach was computationally infeasible.

2.2 Solution Strategy

We implemented a hybrid Blocking + Machine Learning approach.

Approach Type:
🔒 Multi-strategy Blocking + 🤖 Supervised Classifier

Core Innovation:
Instead of comparing every Source 1 record with every Source 2/3 record, we generated candidates through multiple complementary blocking strategies and then used similarity-based machine-learning features to determine whether each candidate pair represented the same entity.

The overall pipeline was:

Raw Records
     ↓
Data Normalization
     ↓
Multiple Blocking Strategies
     ↓
Candidate Pair Generation
     ↓
Similarity Feature Engineering
     ↓
LightGBM Classifier
     ↓
Threshold Optimization
     ↓
Final Entity Matches
3. Candidate Generation (Blocking)

Blocking was the most important scalability component of the solution.

A brute-force comparison would require approximately:

22,774,876,013,799 candidate comparisons

Instead, multiple blocking strategies were combined to generate a much smaller candidate set.

Blocking keys used

The pipeline included:

🔹 Exact normalized business name
🔹 Exact normalized name core
🔹 Exact normalized address
🔹 Name-token blocking
🔹 Address-token blocking
🔹 Name-core + address-token blocking
🔹 Prefix-based blocking
🔹 Prefix + address-number blocking
🔹 Address two-token + number blocking
🔹 Rare-token + address-number blocking
🔹 N-gram blocking
🔹 Phonetic blocking
🔹 Transliteration-based blocking

Country information was incorporated into several blocking keys to reduce irrelevant comparisons.

Candidate volume

The final training blocking pipeline generated:

347,588,284 candidate pairs

compared with approximately:

22.77 trillion brute-force comparisons

This corresponds to approximately:

99.99847% candidate reduction

Blocking Recall

The blocking stage recovered:

6,936,440 / 7,638,365 true pairs

resulting in:

90.81% blocking recall

Preventing candidate explosion

High-frequency blocking keys can produce extremely large Cartesian products.

To control this, we introduced overflow protection and capped individual block-key candidate generation at:

5,000 candidate pairs per key

Large intermediate datasets were also processed using partitioned Parquet files and memory-efficient dataframe operations.

4. Matching Model

After blocking, each candidate pair was converted into similarity features.

Features used
🔤 Name features
Jaro-Winkler similarity
Normalized Levenshtein similarity
Name token Jaccard similarity
Name-core Jaro-Winkler similarity
📍 Address features
Address token Jaccard similarity
Address-number exact-match feature
🌎 Other features
Country exact match
Number of blocking strategies that generated the candidate pair

These features allowed the model to combine multiple weak signals rather than relying on a single string similarity measure.

Model type

LightGBM Gradient Boosted Decision Tree

The matcher was trained using approximately:

259,863 positive examples
1,299,729 negative examples
~1.95 million effective training pairs

Negative examples were downsampled to make model training computationally manageable while maintaining a useful class balance.

Threshold selection

The model generated a matching score for each candidate pair.

Multiple thresholds were evaluated on a validation subset using the challenge's F₀.₅ metric, which places greater emphasis on precision than recall.

The selected threshold was:

0.95

5. Results & Error Analysis
🏆 Final Results
Metric	Result
F₀.₅ Score	0.784
Final Rank	4,521 / 10,650
Teams Submitted	10,650
Blocking Recall	90.81%
Candidate Pairs	347,588,284
Brute-force Search Space	22,774,876,013,799
Candidate Reduction	99.99847%
Common false positives

Potential false positives primarily arise from records sharing common business names, common address tokens, or other frequently occurring attributes.

Because the challenge prioritizes precision through the F₀.₅ metric, candidate generation was deliberately controlled to avoid introducing excessive noisy candidate pairs.

Common false negatives

The missed matches identified during analysis were primarily associated with:

Significant surface differences in business names
Address formatting variations
Shared but insufficiently distinctive address tokens
Variations that did not satisfy any existing blocking key
High-frequency blocking keys where candidate generation was intentionally capped
Records requiring combinations of multiple weak signals

This highlighted an important trade-off between blocking recall and candidate volume.

6. Conclusion

We developed a scalable hybrid entity-resolution system combining normalization, multi-strategy blocking, fuzzy similarity features, and LightGBM classification. The system reduced a 22.77-trillion comparison search space to 347.6 million candidate pairs, achieving 90.81% blocking recall while remaining computationally manageable.

The final submitted solution achieved an F₀.₅ score of 0.784 and ranked 4,521 among 10,650 teams. The project demonstrated the importance of combining data engineering, information retrieval, string similarity, and machine learning for large-scale entity resolution.

Appendix
A. Code Artefacts

The implementation is organized into separate components for normalization, candidate generation, analysis, model training, threshold tuning, and submission generation.

Core pipeline
src/
├── normalize_entities.py
├── build_candidates.py
├── train_matcher.py
├── tune_threshold.py
└── generate_submission.py
Analysis
analysis/
├── analyze_data.py
├── analyze_blocking.py
├── analyze_missed_pairs.py
├── analyze_missed_blocks.py
├── analyze_missed_one_token_number.py
├── analyze_missed_token_frequency.py
├── analyze_number_name_block.py
├── analyze_overflow_cost.py
├── analyze_overflow_strategies.py
├── audit_block_contribution.py
└── marginal_block_audit.py
Utilities
utils/
└── validate_submission.py

The main reproduction flow is:

normalize_entities.py
        ↓
build_candidates.py
        ↓
train_matcher.py
        ↓
tune_threshold.py
        ↓
generate_submission.py

Large datasets, generated candidate partitions, intermediate Parquet files, and other computational artifacts are excluded from the repository.

B. Additional Results
📊 Search-space reduction
Brute Force
22.77 Trillion
       │
       │ 99.99847% reduction
       ▼
Candidate Generation
347.6 Million
       │
       ▼
LightGBM Matching
       │
       ▼
Final F₀.₅
0.784
🔍 Blocking performance
Total True Pairs     : 7,638,365
Recovered            : 6,936,440
Missed               :   701,925

Blocking Recall      : 89.81%
🏆 Competition outcome
Final F₀.₅ : 0.784
Rank       : 4,521 / 10,650

Key takeaway: At this scale, entity resolution is not simply a fuzzy-matching problem. Efficient candidate generation, blocking design, memory management, and the precision–recall trade-off are fundamental to building a practical solution.
