"""One-time Titanic load, cleaning, profiling, and EDA outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

HERE = Path(__file__).parent
OUT = HERE / "output"

def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75])
    return int(((s < q1 - 1.5 * (q3-q1)) | (s > q3 + 1.5 * (q3-q1))).sum())

def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=150); plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    # The module's ONLY network/cache dataset load.
    df = sns.load_dataset("titanic")
    df.to_csv(HERE / "titanic.csv", index=False)
    print(df.info()); print(df.describe(include="all")); print("Shape:", df.shape)
    missing = (df.isna().mean() * 100).loc[lambda s: s > 0].round(2)
    print("Missing percentages:\n", missing)
    # embarked <5%: drop; age 5-30%: median impute; deck ~77%: drop.
    clean = df.drop(columns="deck").dropna(subset=["embarked"]).copy()
    clean["age"] = clean.age.fillna(clean.age.median())
    clean.to_csv(OUT / "cleaned_titanic.csv", index=False)
    fare = {"mean": clean.fare.mean(), "median": clean.fare.median(), "mode": clean.fare.mode().iat[0]}
    for col in ["age", "fare"]:
        fig, axes = plt.subplots(1, 2, figsize=(10,4))
        sns.histplot(clean[col], kde=True, ax=axes[0]); sns.boxplot(x=clean[col], ax=axes[1])
        save(fig, f"{col}_univariate.png")
    corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    corr = clean[corr_cols].corr()
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    strongest = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(2)
    fig, ax = plt.subplots(figsize=(7,5)); sns.heatmap(corr, annot=True, cmap="vlag", center=0, ax=ax); save(fig, "correlation_heatmap.png")
    charts = [("sex", "survived", "survival_by_sex.png"), ("pclass", "survived", "survival_by_class.png")]
    for x, y, name in charts:
        fig, ax = plt.subplots(); sns.barplot(data=clean, x=x, y=y, errorbar=None, ax=ax); save(fig, name)
    fig, ax = plt.subplots(); sns.boxplot(data=clean, x="survived", y="age", ax=ax); save(fig, "age_by_survival.png")
    fig, ax = plt.subplots(); sns.scatterplot(data=clean, x="fare", y="age", hue="survived", alpha=.65, ax=ax); save(fig, "fare_age_survival.png")
    z = clean[["age", "fare"]].apply(lambda s: (s-s.mean())/s.std(ddof=0))
    pd.DataFrame({"before_mean":clean[["age","fare"]].mean(), "before_std":clean[["age","fare"]].std(ddof=0), "z_mean":z.mean(), "z_std":z.std(ddof=0)}).to_csv(OUT / "zscore_check.csv")
    sex_rates = clean.groupby("sex").survived.mean().round(3).to_dict()
    class_rates = clean.groupby("pclass").survived.mean().round(3).to_dict()
    mask_rates = {f"{s}_{p}": round(clean.loc[(clean.sex==s) & (clean.pclass==p),"survived"].mean(),3) for s in ["female","male"] for p in [1,2,3]}
    report = f'''# EDA report

Original shape: {df.shape}. Missing percentages: {missing.to_dict()}. `embarked` is {missing['embarked']:.2f}% missing (<5%), so its rows are dropped. `age` is {missing['age']:.2f}% (5–30%), so it is median-imputed. `deck` is {missing['deck']:.2f}% missing, so imputation is unreliable and the column is dropped.

Age IQR outliers: {iqr_outliers(clean.age)}; fare IQR outliers: {iqr_outliers(clean.fare)}. Fare mean/median/mode are {fare['mean']:.2f}/{fare['median']:.2f}/{fare['mode']:.2f}; mean > median > mode indicates a right-skewed distribution. `zscore_check.csv` shows age and fare have approximately mean 0 and standard deviation 1 after exploratory z-score standardization; modelling scales separately after splitting.

Survival rate by sex: {sex_rates}. By pclass: {class_rates}. Boolean masking by sex and pclass: {mask_rates}. The heatmap contains exactly survived, pclass, age, sibsp, parch, fare; its strongest absolute off-diagonal pairs are {[(a,b,round(v,3)) for (a,b),v in strongest.items()]}. They show association, not causation.

## Four-chart story

1. `survival_by_sex.png`: the sex gap makes sex an important predictive candidate.
2. `survival_by_class.png`: first-class passengers survive more often, suggesting class-related advantage.
3. `age_by_survival.png`: the overlapping age distributions show age alone cannot decide survival.
4. `fare_age_survival.png`: survivors occur more often at higher fares, but overlap motivates multifeature modelling.
'''
    (OUT / "EDA_REPORT.md").write_text(report, encoding="utf-8")

if __name__ == "__main__": main()
