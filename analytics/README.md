# Analytics pipeline

Run `python analytics/01_eda.py` once online, then commit its `titanic.csv`
offline fallback. Run `python analytics/02_modeling.py`; it reads that same CSV
and never calls Seaborn's loader. The `output/` folder contains charts, reports,
metrics, and the complete saved pipeline. Read `EDA_REPORT.md` and
`MODELING_REPORT.md` after execution: all numeric findings are generated from
the run rather than manually copied.
