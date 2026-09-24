# Zepto Data & AI Platform

This repository is a single capstone project with three modules at the root level:

- `/data_pipeline` for scraping, cleaning, and normalizing book data
- `/analytics` for exploratory analysis and Titanic modeling
- `/support_assistant` for the offline-first support assistant

## Repository structure

```text
Capstone_Project_masai/
├── README.md
├── requirements.txt
├── data_pipeline/
│   ├── README.md
│   ├── run_pipeline.py
│   └── output/
│       ├── books.db
│       ├── clean_books.csv
│       ├── join_sql.csv
│       ├── join_pandas_merge.csv
│       ├── join_validation.txt
│       ├── queries.sql
│       └── query_outputs.json
├── analytics/
│   ├── README.md
│   ├── 01_eda.py
│   └── 02_modeling.py
├── support_assistant/
│   ├── README.md
│   ├── Dockerfile
│   ├── main.py
│   └── docs/
└── work/
```

## Setup

This project uses one consolidated root-level requirements file: `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run each module end-to-end

### 1) Data pipeline

```bash
python data_pipeline/run_pipeline.py
```

This scrapes the first three categories, cleans and normalizes the data, creates the SQLite database, and saves SQL outputs in `/data_pipeline/output`.

### 2) Analytics module

```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```

This performs exploratory analysis and builds the Titanic prediction model.

### 3) Support assistant

```bash
cd support_assistant
python -m uvicorn main:app --reload --port 7860
```

This runs the assistant API locally.

## Git workflow for submission

Before pushing to GitHub, create a feature branch, make at least two commits on it, and merge back into `main` so the repository history clearly shows the expected branch/merge activity.

```bash
git checkout -b feature/capstone
git add .
git commit -m "Add data pipeline module"
git add .
git commit -m "Complete analytics and assistant modules"
git checkout main
git merge --no-ff feature/capstone
```

## Design decisions

### Data pipeline

- Uses a fixed conversion of 1 GBP = 105.50 INR.
- Imputes missing numeric values using the median strategy.
- Drops rows with missing title/category because those values are required for relational joins and analysis.
- Stores the cleaned dataset in SQLite with normalized `categories` and `books` tables.
- Validates SQL output against pandas output to confirm consistency between the relational and in-memory joins.

### Analytics

- Uses a structured exploratory workflow to inspect the Titanic dataset before modeling.
- Builds a reproducible machine-learning pipeline for prediction and evaluation.
- Keeps notebook-style or script-based analysis readable and easy to rerun.

### Support assistant

- Uses a lightweight local application pattern suitable for offline-first or low-resource demonstration.
- Keeps the assistant logic modular and easy to containerize or run locally with Uvicorn.
- Stores supporting documentation in the module folder so setup and behavior are explained in text rather than screenshots or presentation files.

## Module documentation

- Data pipeline: [data_pipeline/README.md](data_pipeline/README.md)
- Analytics: [analytics/README.md](analytics/README.md)
- Support assistant: [support_assistant/README.md](support_assistant/README.md)
