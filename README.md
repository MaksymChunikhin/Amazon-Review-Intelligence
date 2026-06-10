# Amazon Review Intelligence

**English** | [Русский](README.ru.md)

> **TL;DR.** End-to-end NLP project: 500k Amazon Home & Kitchen reviews → sentiment classification
> (best model: fine-tuned **DistilBERT, macro-F1 0.744**, beating a tuned TF-IDF baseline at 0.717),
> 12 LDA topics with pain-point analytics, rating-vs-text mismatch detection, and a bilingual
> Streamlit dashboard with live prediction. Validation is strict throughout: one stratified
> train/test split shared by all five models, tuning via CV on train only, limitations documented.
> Notebooks in [notebooks/](notebooks/) are bilingual (EN/RU).

End-to-end NLP project: from a raw JSONL with hundreds of thousands of Amazon reviews to an interactive bilingual dashboard with sentiment analysis, topic modeling, and live prediction.

**Problem.** An Amazon seller (Home & Kitchen category) has hundreds of thousands of reviews. The star rating shows *how* satisfied a customer is, but not *what exactly* went wrong or where the hidden dissatisfaction is (a 5★ with an irritated text). No one can read all of it manually.

**Solution.** A pipeline that turns raw text into analytics: classifies sentiment (3 classes), extracts 12 topics and customer pain points, finds rating-vs-text mismatches, and serves all of it in a dashboard with real-time prediction. Five models compared fairly (same `train_test_split`, same labeling); the best one is a fine-tuned **DistilBERT**.

**Results.**
- **DistilBERT: macro-F1 0.744, neutral recall 0.69** — beats both the tuned baseline (0.717) and the classic TF-IDF models.
- Pain points localized: *Price, Value & Delivery Issues* — **41.8% negative**, *Complaints & Malfunctions* — 34.1%.
- Mismatch analysis: **11.3% of all 1★ reviews** have positive text — a signal of mistakenly lowered ratings.
- Interactive **Streamlit dashboard** (RU/EN) with 5 tabs and a live predictor.

![Model comparison by macro-F1](assets/macro_f1.png)

> ### Honest about limitations
> Sentiment is derived from the **star rating** (weak supervision), not from manual text labeling: 1–2★ → negative, 3★ → neutral, 4–5★ → positive. This yields free labels for 460k reviews, but the price is noise — especially the **neutral** class (3★), which is poorly separable from its neighbors by text alone and drags down recall for every model. That is why the key metric is **macro-F1**, not accuracy (the dataset is ~82% positive). And the "rating ≠ text" mismatch cases are a mix of *real* discrepancies and *model errors*; the dashboard lets you filter them by confidence, but they cannot be fully separated.

---

## Dashboard

Interactive Streamlit dashboard with a bilingual (RU/EN) interface and five tabs. See [Interactive dashboard](#interactive-dashboard-streamlit) below for how to run it.

**Overview** — KPIs and distributions of ratings and sentiment:
![Overview](assets/overview.jpg)

**Topics & Pain Points** — topic × sentiment heatmap, the most negative topics:
![Topics & Pain Points](assets/topics.jpg)

**Mismatch Explorer** — reviews where the text contradicts the stars (5★ + negative, 1★ + positive):
![Mismatch Explorer](assets/mismatch.jpg)

**Live Predictor** — type a review → sentiment, confidence, and topic in real time:
![Live Predictor](assets/predictor.jpg)

**Trends** — review volume and sentiment shares by month:
![Trends](assets/trends.jpg)

---

## What's inside

The analysis is split into three notebooks in `notebooks/`, executed in order: **01** prepares the data
(`data/reviews_clean.parquet`), **02** trains the sentiment models and fixes the train/test split
(`data/split.parquet`), **03** builds topics and business insights on the same data and split.
All markdown descriptions in the notebooks are bilingual (EN + RU).

| Section | Contents |
|---------|----------|
| **1.1 Data Ingestion** | Chunked reading of a large JSONL + uniform 500k-review sample via reservoir sampling, year filter (≥ 2019), parquet cache |
| **1.2 Data Cleaning** | Duplicate removal (by row and by text), filtering out too-short reviews |
| **1.3 NLP Preprocessing** | Text cleaning (HTML, contractions, punctuation), merging `title + text`, stop-word removal |
| **1.4 EDA** | Rating distribution and class imbalance; star-based sentiment labeling (weak supervision); word and bigram frequency analysis per class; trends over time |
| **2.1 Sentiment Analysis** | 3 baseline models on TF-IDF: Logistic Regression, **LinearSVC (calibrated)**, LightGBM; error analysis, comparison by accuracy / macro-F1 / weighted-F1 |
| **2.2 Deployment** | End-to-end Pipeline `raw text → cleaning → model`, saving and verification |
| **2.3 LinearSVC tuning** | `RandomizedSearchCV` over `C` and TF-IDF parameters, optimized for macro-F1; tuned baseline (macro-F1 0.657 → 0.717, neutral recall 0.17 → 0.46) |
| **2.4 Transformer (DistilBERT)** | Fine-tuning DistilBERT on raw text (GPU, fp16, weighted loss) — **the project's best model** (macro-F1 0.744, neutral recall 0.69) |
| **3.1 Topic Modeling (LDA)** | LDA (12 topics) on CountVectorizer, business labeling of topics, topic × sentiment, topic trends over time |
| **3.2 Mismatch Analysis** | Discrepancies between star rating and predicted sentiment (with a confidence filter) |
| **3.3 Insight Fusion** | Most negative topics, helpful votes and verified purchase by topic, sentiment trends |
| **3.4 Dashboard** | Summary visualizations |
| **3.5 BERTopic vs LDA** | Topic modeling on semantic embeddings (MiniLM) as a comparison with LDA; exploratory section, deliberate choice of LDA for the dashboard |

---

## Project structure

```
Amazon Review Intelligence/
├── notebooks/
│   ├── 01_data_eda.ipynb              # ingestion, cleaning, preprocessing, EDA
│   ├── 02_sentiment_models.ipynb      # baseline models, deployment, tuning, DistilBERT
│   └── 03_topics_insights.ipynb       # LDA, mismatch, insights, BERTopic vs LDA
├── app.py                             # interactive Streamlit dashboard
├── build_dashboard_data.py            # rebuilds the enriched parquet for the dashboard
├── nlp_utils.py                       # shared text-cleaning functions (notebooks + dashboard)
├── distilbert_utils.py                # DistilBERT inference for the dashboard
├── data/
│   ├── Home_and_Kitchen.jsonl         # raw data (not in git)
│   ├── reviews_processed.parquet      # sample cache (not in git)
│   ├── reviews_clean.parquet          # cleaned dataset from 01 (not in git)
│   ├── split.parquet                  # fixed train/test split from 02 (not in git)
│   └── reviews_dashboard.parquet      # enriched data for the dashboard (not in git)
├── models/                            # trained artifacts (not in git)
│   ├── log_reg_pipeline.joblib
│   ├── svc_pipeline.joblib
│   ├── lightgbm_pipeline.joblib
│   ├── lda_model.joblib
│   ├── topic_vectorizer.joblib
│   ├── final_sentiment_model.joblib   # deployment model (raw text → sentiment)
│   ├── svc_tuned_pipeline.joblib      # tuned LinearSVC (section 2.3)
│   └── distilbert_sentiment/          # fine-tuned DistilBERT (section 2.4)
├── requirements.txt
├── .gitignore
├── README.md                          # this file (English)
└── README.ru.md                       # Russian version
```

---

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / WSL
pip install -r requirements.txt
jupyter notebook notebooks/     # run in order: 01 → 02 → 03
```

> **Important:** the artifacts in `models/*.joblib` are sensitive to the scikit-learn version
> (pickle is not guaranteed across versions: you may see `NotFittedError` or
> `InconsistentVersionWarning`). Use the versions from `requirements.txt`,
> or delete `models/*.joblib` — the notebooks will retrain the models from scratch.

Data: [Amazon Reviews 2023 (McAuley Lab, UCSD)](https://amazon-reviews-2023.github.io/), *Home & Kitchen* category.
Download `Home_and_Kitchen.jsonl` and put it in `data/`. If `reviews_processed.parquet` is present,
the ingestion step is skipped.

---

## Interactive dashboard (Streamlit)

```bash
# 1. rebuild the enriched data once (sentiment, topics, model predictions)
python build_dashboard_data.py        # creates data/reviews_dashboard.parquet (requires a GPU)
# 2. launch the dashboard
streamlit run app.py                  # opens at http://localhost:8501
```

Sentiment predictions in the dashboard are made by the **fine-tuned DistilBERT** (the best model from section 2.4),
topics — by LDA. The dashboard loads the prebuilt parquet and models and retrains nothing. Tabs:

| Tab | What it shows |
|---|---|
| **📊 Overview** | KPIs (count, avg. rating, % positive/verified) + rating and sentiment distributions |
| **🏷️ Topics & Pain Points** | Topic × sentiment heatmap, most negative topics, helpful votes by topic, example reviews |
| **📈 Trends** | Review volume and sentiment shares by month |
| **⚠️ Mismatch Explorer** | Discrepancies between rating and DistilBERT-predicted sentiment (confidence slider) |
| **🔮 Live Predictor** | Type your own review → sentiment + confidence (DistilBERT) + topic in real time |

> `build_dashboard_data.py` replicates the cleaning and labeling from the notebook, runs **DistilBERT**
> and LDA over all reviews, and saves a slim parquet — which is why the dashboard loads instantly.
> DistilBERT inference requires a GPU build of torch, so on Windows the script is run from PowerShell.
> Text preprocessing lives in `nlp_utils.py`, DistilBERT inference in `distilbert_utils.py`.

> ⚠️ **Coverage differs from the notebook.** The dashboard computes sentiment predictions and topics
> over **all** ~460k reviews, whereas in the notebook predictions were made on the test set
> (20%) and topic modeling on a 200k sample. Therefore the absolute mismatch counts in the
> dashboard will be higher than in the notebook (the shares remain comparable).

---

## Target variable

Sentiment is derived from the star rating (weak supervision):

| Rating | Sentiment |
|--------|-----------|
| 1–2    | negative  |
| 3      | neutral   |
| 4–5    | positive  |

The dataset is heavily imbalanced (~82% positive), so the key metric is **macro-F1**, not accuracy.

---

## Key results

Sentiment classification (3 classes, test set of 92,140 reviews):

| Model | accuracy | macro F1 | weighted F1 |
|---|---|---|---|
| Logistic Regression | 0.862 | **0.694** | 0.879 |
| **LinearSVC (calibrated)** | **0.901** | 0.657 | **0.886** |
| LightGBM | 0.859 | 0.696 | 0.877 |

- **Deployment baseline — LinearSVC** (calibrated): best accuracy (0.90) and weighted-F1
  (0.886) among the three baselines. Calibration via `CalibratedClassifierCV` provides `predict_proba`,
  so SVC serves as a single model both for confidence analysis and for lightweight CPU deployment
  (`final_sentiment_model.joblib`). The price is low recall on the **neutral** class (0.17) and
  the worst macro-F1 of the three (0.657).
- **The project's best model — DistilBERT** (section 2.4, macro-F1 **0.744**, neutral recall 0.69);
  it is the one making predictions in the dashboard. The roles are split deliberately: LinearSVC is a fast
  CPU baseline for deployment, DistilBERT is maximum quality on GPU for analytics.
- The hardest class is **neutral** (rating 3): it gets confused with positive/negative, which is
  expected for weak-labeled data.
- Topic modeling (12 topics) surfaces customer pain points: the most negative topics are
  *Price, Value & Delivery Issues* (41.8% negative) and *Complaints & Malfunctions*
  (34.1%); the most positive are *Gifts & Recommendations* and *Ease of Use & Cleaning* (>93%).
- Mismatch analysis: "1★ + positive" occurs more often (11.3% of all 1★) than "5★ + negative"
  (0.5% of all 5★) — the model is biased toward positive. A confidence filter (≥0.70) cuts off ~70%
  as likely model errors, leaving the real discrepancies.

> ⚠️ LDA does not guarantee stable topic indices across retrainings: the business labels
> in the notebook (`topic_labels`) were matched by top words and, after retraining the model,
> need to be re-verified against the `print_topics` output.

---

## Experiment: tuned baseline vs Transformer (DistilBERT)

Sections **2.3–2.4 of notebook 02** test whether a contextual transformer beats the classic baseline —
and how much honest tuning of the baseline itself helps. All comparisons are fair: the **same**
`train_test_split` (`random_state=42`), the same labeling, the same test set (92,140 reviews).

- **LinearSVC (tuned)** (section 2.3): `RandomizedSearchCV` over `C` and TF-IDF parameters, optimized
  for macro-F1, on the full train set (CPU).
- **DistilBERT** (section 2.4): raw text (`title + text`), GPU (RTX 3090, fp16, 2 epochs, ~16 min),
  **weighted loss** against the ~82/11/7 imbalance.

| Model | accuracy | macro F1 | weighted F1 | recall (neutral) |
|---|---|---|---|---|
| Logistic Regression | 0.862 | 0.694 | 0.879 | 0.60 |
| LinearSVC (calibrated, **untuned**) | **0.901** | 0.657 | 0.886 | 0.17 |
| LightGBM | 0.859 | 0.696 | 0.877 | 0.62 |
| LinearSVC (**tuned**, `C≈0.06`) | 0.896 | 0.717 | 0.899 | 0.464 |
| **DistilBERT (fine-tuned)** | 0.889 | **0.744** | **0.903** | **0.692** |

- **Tuning a heavily underrated baseline:** LinearSVC macro-F1 went up **0.657 → 0.717 (+6.0 pp)**,
  neutral recall **0.17 → 0.46**. The main driver is strong regularization (`C≈0.06` vs the default `1.0`).
- **DistilBERT still wins** on both key metrics: macro-F1 **0.744** vs 0.717 and
  especially neutral recall **0.69** vs 0.46. Context gives a robust advantage on the hardest class.
- **The advantage is statistically significant:** a bootstrap of the macro-F1 difference (1000 test resamples)
  gives +0.028 with a 95% CI of **[+0.023, +0.032]**, p < 0.001 — the gap is not explained by sampling noise.
- **The cost of `max_length=128`:** the limit truncates **11.9%** of test reviews (median length — 44 tokens,
  95th percentile — 198), i.e. for the vast majority of reviews the model sees the full text.
- Takeaway: the tuned baseline closes most of the gap, but DistilBERT still leads on the key metrics —
  thanks to context, which TF-IDF models have no access to.

> DistilBERT requires a GPU build of torch (the PyPI build for Linux already includes CUDA) + `accelerate`.
> The code is in `notebooks/02_sentiment_models.ipynb`
> (sections 2.3–2.4); models are cached in `models/distilbert_sentiment/` and `models/svc_tuned_pipeline.joblib`
> (artifacts are not versioned); if present, the cells load them, otherwise they train/search from scratch.

---

## Bonus: BERTopic vs LDA (section 3.5)

A separate exploratory section compares LDA with **BERTopic** — topic modeling on semantic embeddings (MiniLM, GPU). On a 50k-review sample:

- BERTopic discovered about **65 topics** on its own versus the manually set **12** of LDA, and they are more specific: where LDA gives a single "Bedding & Sleep", BERTopic separates pillows, mattresses, sheets, and blankets.
- About **46% of reviews** ended up in the "Other" group — short generic reviews ("great", "works well") with no distinct topic. LDA would have force-assigned them.
- The price: heavier (GPU + `bertopic`, `sentence-transformers`, `umap-learn`, `hdbscan`) and non-deterministic (UMAP introduces randomness).

**Engineering decision:** the dashboard deliberately keeps **LDA** — fast, reproducible, instantly labels all ~460k reviews, and fits into 12 interpretable topics. BERTopic is more precise and specific, but 65 topics and 46% "Other" would require redesigning the analytics and recomputing embeddings on GPU over the whole dataset. The *fast + reproducible* vs *precise + heavy* trade-off was resolved in favor of the former; BERTopic is documented as the direction to move toward when fine-grained, product-level segmentation is needed.

---
