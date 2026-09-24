# Data pipeline

Run `python data_pipeline/run_pipeline.py`. It scrapes the first three catalogue
categories, including their pagination, so it obtains at least 60 books. It
stores a clean CSV, `output/books.db`, SQL-query results, and the SQL/pandas
join outputs.

Cleaning converts prices to floats, rating words to integers, and availability
to booleans. Unexpected numeric values are median-imputed; rows without a title
or category are dropped because those fields are relational identifiers and
cannot be defensibly inferred. `price_inr = price_gbp * 105.50`; this is the
assignment's fixed baseline, not a live exchange rate.

The normalized SQLite schema has `categories` and `books`; `books.category_id`
is a foreign key. `query_outputs.json` includes WHERE, ORDER BY, LIMIT,
DISTINCT, BETWEEN, IN, and JOIN examples. `join_sql.csv` and
`join_pandas_merge.csv` demonstrate the same join through `pd.read_sql` and
in-memory `pd.merge`.
