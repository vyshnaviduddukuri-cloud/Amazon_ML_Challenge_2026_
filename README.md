# 🔗 Large-Scale Entity Resolution & Record Linkage

> 🚀 **Scalable entity matching system for linking millions of business records across multiple data sources using intelligent blocking, fuzzy similarity features, and LightGBM.**

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Polars-Fast%20Data%20Processing-orange)](https://pola.rs/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Machine%20Learning-green)](https://lightgbm.readthedocs.io/)
[![Parquet](https://img.shields.io/badge/Storage-Parquet-yellow)](https://parquet.apache.org/)

---

## 🧩 Overview

Entity resolution is the process of determining which records from different datasets refer to the **same real-world entity**.

In this challenge, millions of business records were distributed across three different sources:

* 🗂️ **Source 1** — Reference entities
* 🗂️ **Source 2** — Candidate entities
* 🗂️ **Source 3** — Candidate entities

The records could contain variations in:

* 🏢 Business names
* 📍 Addresses
* 🌎 Countries
* ✍️ Abbreviations
* 🔤 Formatting
* 🔀 Word ordering
* 🌐 Transliteration
* 🧹 Noisy or incomplete data

The goal was to identify **all matching entities across sources while maintaining high precision and recall**.

---

## 🎯 Challenge

A brute-force comparison between every Source 1 record and every Source 2/3 record would require approximately:

### 💥 22.77 Trillion comparisons

This is computationally impractical.

So instead of comparing everything with everything, this project uses a **multi-stage entity-resolution pipeline** to drastically reduce the search space while preserving as many true matches as possible.

---

# 🏗️ Architecture

```text
                    📂 Raw Datasets
                          │
                          ▼
                 🧹 Data Normalization
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Name Processing        Address Processing
              │                       │
              └───────────┬───────────┘
                          ▼
                 🔒 Candidate Blocking
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          Exact Match   Token Match   Prefix Match
              │           │           │
              ├───────────┼───────────┤
              ▼           ▼           ▼
        N-Gram / Phonetic / Number / Transliteration
                          │
                          ▼
                 🎯 Candidate Pairs
                          │
                          ▼
              🧮 Feature Engineering
                          │
                          ▼
                   🤖 LightGBM
                          │
                          ▼
                📊 Probability Score
                          │
                          ▼
                 🎚️ Threshold Tuning
                          │
                          ▼
                 🔗 Final Entity Matches
```

---

# ⚡ Key Achievement

The most important challenge was reducing the search space without losing too many genuine matches.

| Metric                       |                   Result |
| ---------------------------- | -----------------------: |
| 🔢 Brute-force comparisons   |       **22.77 Trillion** |
| 🎯 Candidate pairs generated |        **347.6 Million** |
| 📉 Candidate reduction       |            **99.99847%** |
| 🔍 Blocking recall           |               **90.81%** |
| 🤖 ML matcher                |             **LightGBM** |
| 🏆 Final F0.5                |                **0.784** |
| 🥇 Final ranking             | **4,521 / 10,650 teams** |

> 💡 The pipeline reduced a **22.77 trillion-pair search space to 347.6 million candidate pairs**, making large-scale matching computationally feasible.

---

# 🧹 1. Data Normalization

Raw business records were transformed into matching-friendly representations.

### Name normalization

The pipeline handles:

* Unicode normalization
* Case normalization
* Punctuation removal
* Whitespace normalization
* `&` → `and`
* Legal suffix handling
* Country-aware normalization

Example:

```text
"ABC Pvt. Ltd."
        ↓
"abc pvt ltd"
        ↓
"abc"
```

### Address normalization

Addresses are normalized to handle:

* Abbreviations
* Punctuation
* Case differences
* Whitespace
* Country-specific variations

Example:

```text
"12, MG Road, Bangalore"
        ↓
"12 mg road bangalore"
```

---

# 🔒 2. Intelligent Blocking

Comparing every record against every other record is impossible at this scale.

The project therefore uses **multiple blocking strategies**.

### Blocking strategies include:

* 🔹 Exact normalized name
* 🔹 Exact normalized name core
* 🔹 Exact normalized address
* 🔹 Name-token blocking
* 🔹 Address-token blocking
* 🔹 Prefix blocking
* 🔹 Rare-token + address-number blocking
* 🔹 Phonetic blocking
* 🔹 N-gram blocking
* 🔹 Transliteration blocking
* 🔹 Prefix + number rescue blocks
* 🔹 Address-token + number blocking

Each block creates a smaller set of plausible candidate matches.

### 🛡️ Overflow protection

High-frequency tokens can generate enormous Cartesian products.

To prevent candidate explosion, blocks use configurable limits such as:

```python
MAX_PAIRS_PER_BLOCK_KEY = 5000
```

This keeps candidate generation scalable.

---

# 🧮 3. Feature Engineering

Each candidate pair is transformed into numerical similarity features.

### Name features

* 🔤 Jaro-Winkler similarity
* 🔤 Normalized Levenshtein similarity
* 🔤 Token Jaccard similarity

### Address features

* 📍 Address token Jaccard similarity
* 🔢 Address-number exact match

### Additional features

* 🌎 Country exact match
* 🔗 Number of blocking strategies that produced the candidate

These features allow the model to distinguish between strong and weak candidate matches.

---

# 🤖 4. Machine Learning Matcher

The candidate pairs are scored using **LightGBM**.

The model learns the relationship between similarity features and whether two records represent the same entity.

### Training

The training pipeline used approximately:

```text
Positive pairs       : 259K
Negative pairs       : 1.30M
Effective train set  : ~1.95M
```

Negative downsampling was used to keep training computationally manageable.

---

# 🎚️ 5. Threshold Optimization

The LightGBM model produces a probability-like match score.

Instead of blindly using `0.5`, multiple thresholds were evaluated using the challenge's **F0.5 metric**.

The selected validation threshold was:

```text
Threshold = 0.95
```

F0.5 was prioritized because precision was more heavily weighted than recall.

---

# 📊 Final Results

The final submitted system achieved:

### 🏆 F0.5: **0.784**

### 📈 Leaderboard

**Rank: 4,521 / 10,650 teams**

This placed the solution ahead of thousands of participating teams while operating under strict computational constraints.

---

# 💻 Technology Stack

| Technology           | Purpose                          |
| -------------------- | -------------------------------- |
| 🐍 Python            | Main implementation              |
| ⚡ Polars             | Large-scale dataframe processing |
| 🗄️ Parquet          | Partitioned candidate storage    |
| 🤖 LightGBM          | Candidate matching model         |
| 🔤 String Similarity | Entity comparison                |
| 🧮 Jaro-Winkler      | Name similarity                  |
| 📐 Levenshtein       | Edit-distance similarity         |
| 🔗 Token Jaccard     | Token-level similarity           |
| 🧩 Blocking          | Search-space reduction           |

---

# 📁 Project Structure

```text
entity-resolution/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 🧹 normalize_entities.py
├── 🔒 build_candidates.py
├── 🤖 train_matcher.py
├── 🎚️ tune_threshold.py
├── 📤 generate_submission.py
│
├── 🔍 analyze_missed_pairs.py
├── 📊 analyze_blocking.py
│
├── 📁 docs/
│   ├── architecture.md
│   └── results.md
│
└── 📁 dataset/
    └── README.md
```

> ⚠️ Large datasets, generated candidate partitions, model artifacts, and submission files are intentionally excluded from the repository.

---

# 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd entity-resolution
```

### 2️⃣ Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Prepare datasets

Place the challenge datasets under:

```text
dataset/
├── train/
└── test/
```

See `dataset/README.md` for the expected structure.

### 5️⃣ Normalize the data

```bash
python3 normalize_entities.py
```

### 6️⃣ Generate candidates

```bash
python3 build_candidates.py
```

### 7️⃣ Train the matcher

```bash
python3 train_matcher.py
```

### 8️⃣ Tune the threshold

```bash
python3 tune_threshold.py
```

### 9️⃣ Generate predictions

```bash
python3 generate_submission.py
```

---

# 🧠 What This Project Demonstrates

This project goes beyond basic machine learning.

It demonstrates practical experience with:

* 🏗️ Large-scale data processing
* 🔗 Entity resolution
* 🔒 Blocking strategies
* 🧹 Data normalization
* 🔤 Fuzzy string matching
* 🧮 Feature engineering
* 🤖 Gradient-boosted machine learning
* 📊 Precision/recall optimization
* 💾 Memory-efficient processing
* ⚡ Partitioned Parquet workflows
* 📈 Evaluation and threshold tuning
* 🛠️ Building scalable pipelines under resource constraints

---

# 💡 Key Learning

The biggest lesson from this project was that **entity resolution is not simply a fuzzy-string-matching problem**.

At millions of records, the primary challenge becomes:

> **How do you find the small set of plausible candidates without comparing everything with everything?**

That makes **blocking strategy, candidate reduction, memory management, and precision/recall trade-offs** just as important as the machine-learning model itself.

---

# 🏁 Final Takeaway

This project combines:

**Data Engineering ⚙️ + NLP 🔤 + Machine Learning 🤖 + Information Retrieval 🔎 + Scalable Computing 🚀**

into one end-to-end entity-resolution system.

> **22.77T possible comparisons → 347.6M candidates → 90.81% blocking recall → F0.5 0.784 → Rank 4521/10650 🏆**

---

## 👩‍💻 Author

**Vyshnavi Dhudhukuri**

B.Tech — Computer Science & Business Systems

Interested in:

`Software Engineering` • `Data Engineering` • `Machine Learning` • `AI` • `DSA`

---

⭐ If you found this project interesting, consider starring the repository!
