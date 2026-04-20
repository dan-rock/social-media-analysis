# Social Media Behavioral Profiling Pipeline
**NLP · Regression · Topic Modeling · Python**

End-to-end NLP pipeline analyzing 10,000+ scraped social media posts to extract behavioral signals — framing a scalable approach to content-based user risk assessment applicable to platform safety contexts.

---

## Project Structure

```
social-media-profiling/
├── 01_data_collection.ipynb      # web scraping & CSV export
├── 02_engagement_analysis.ipynb  # Linear regression on engagement signals
├── 03_nlp_analysis.ipynb         # NLP preprocessing, NER & topic modeling
├── trump_tweets.csv              # Generated dataset (run notebook 01 first)
└── README.md
```

---

## Pipeline Overview

### 01 — Data Collection
- Scrapes tweet data from [The Trump Archive](https://www.thetrumparchive.com/)
- Collects 10,000 tweets with fields: `text`, `favorites`, `retweets`, `date`, `isDeleted`, `isRetweet`, `device`
- Exports to `trump_tweets.csv`

### 02 — Engagement Analysis
- Filters to original, non-deleted tweets
- Explores distributions of favorites and retweets (histograms, boxplots)
- Fits baseline linear regression; identifies non-linearity (fan-shaped residuals)
- Applies log, sqrt, and cube root transformations — log transform selected as optimal
- **Result:** Log-transformed regression achieves **R² ≈ 0.88**, quantifying content amplification dynamics

### 03 — NLP Analysis
- **Preprocessing:** URL removal, punctuation stripping, tokenization, stop word removal, lemmatization
- **Word Frequency:** Term counts across the full corpus; top 30 most-used words visualized
- **NER (spaCy):** Extracts and categorizes named entities into organizations, places, and people/things
- **Topic Modeling (LDA):** Surfaces 6 latent discourse clusters from the tweet corpus

---

## Key Results

| Analysis | Method | Result |
|---|---|---|
| Engagement modeling | Log-linear regression | R² ≈ 0.88 (favorites → retweets) |
| Top entity — org | spaCy NER | Senate, FBI |
| Top entity — place | spaCy NER | America, China |
| Top entity — person | spaCy NER | Biden |
| Topic clusters | LDA (6 topics) | Political opponents, foreign policy, media, economy |

---

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy \
            nltk spacy gensim pyLDAvis transformers requests selenium

python -m spacy download en_core_web_sm
```

Run notebooks in order: `01` → `02` → `03`

> `02` and `03` both depend on `trump_tweets.csv` produced by `01`.
