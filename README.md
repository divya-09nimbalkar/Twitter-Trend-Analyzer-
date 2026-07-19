
---

# Twitter Trend Analyzer 

##  Overview
The **Twitter Trend Analyzer** is a data engineering and machine learning pipeline designed to:
- Extract tweets using the Twitter API (via Tweepy)
- Transform and clean text data for analysis
- Train sentiment or trend detection models
- Predict sentiment on new tweets
- Visualize results in an interactive **Streamlit dashboard**

---

##  Project Structure
```
Twitter_Trend_Analyzer/
│
├── data/
│   ├── raw/                # raw tweets or API dumps
│   └── processed/          # cleaned datasets
│
├── docs/                   # documentation
├── models/                 # trained ML models
├── notebooks/              # Jupyter notebooks
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── main.py             # pipeline entry point
│   ├── extract.py          # fetch tweets via API
│   ├── transform.py        # clean text, tokenize, sentiment
│   ├── load.py             # save processed data
│   ├── train.py            # train ML models
│   ├── predict.py          # predict sentiment/trends
│   └── dashboard.py        # Streamlit dashboard
├── tests/
│   ├── test_extract.py
│   ├── test_transform.py
│   └── test_train.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

##  Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/divya-09nimbalkar/Twitter-Trend-Analyzer-.git
   cd Twitter-Trend-Analyzer-
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

### 1. Prepare Data
- Place raw tweets in `data/raw/tweets.csv`  
- Or fetch tweets live using the Twitter API in `src/extract.py`
- Or import a reviewed Xquik export:

```bash
python -m src.main --xquik-export path/to/xquik-tweets.json
```

The [Xquik](https://xquik.com) import path accepts saved CSV, JSON, and JSONL
tweet results. It maps fields such as `text`, `createdAt`, and nested
`author.username` into the project schema, then classifies them with the
existing sentiment model.

### 2. Run ETL Pipeline
```bash
python -m src.main
```
This will:
- Load raw tweets
- Clean and preprocess them
- Save processed data to `data/processed/tweets_clean.csv`
- Train a sentiment model (`models/sentiment_model.pkl`)

### 3. Run Dashboard
Start Streamlit:
```bash
python -m streamlit run src/dashboard.py
```
Upload a CSV of tweets and view sentiment predictions.

---

##  Exploration Notebook
Open:
```bash
jupyter notebook notebooks/exploration.ipynb
```
This notebook demonstrates:
- Data loading
- Text preprocessing
- Exploratory plots
- Sentiment distribution

---

##  Testing
Run unit tests:
```bash
pytest tests/
```

---

## Data Schema

Training and dashboard uploads should include tweet text. These columns are
recognized:

- Text: `text`, `full_text`, `tweet_text`, or `content`
- Date: `date`, `created_at`, `createdAt`, or `timestamp`
- User: `user`, `username`, `author_username`, or `screen_name`
- Sentiment: `sentiment`, `label`, or `predicted_sentiment`

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.
