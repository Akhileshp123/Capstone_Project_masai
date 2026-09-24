"""Leakage-safe Titanic classification, tuning, regression, persistence."""
from pathlib import Path
import joblib, pandas as pd, matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

HERE=Path(__file__).parent; OUT=HERE/"output"
NUM=["pclass","age","sibsp","parch","fare"]; CAT=["sex","embarked"]; FEATURES=NUM+CAT
def prep():
    return ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUM), ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),CAT)])
def score(name, pipe, xt, yt):
    p=pipe.predict(xt); prob=pipe.predict_proba(xt)[:,1]
    fig,axes=plt.subplots(1,2,figsize=(9,4)); ConfusionMatrixDisplay.from_predictions(yt,p,ax=axes[0]); RocCurveDisplay.from_predictions(yt,prob,ax=axes[1]); fig.tight_layout(); fig.savefig(OUT/f"{name}_evaluation.png",dpi=150); plt.close(fig)
    return {"model":name,"accuracy":accuracy_score(yt,p),"precision":precision_score(yt,p),"recall":recall_score(yt,p),"f1":f1_score(yt,p),"auc":roc_auc_score(yt,prob)}
def main():
    OUT.mkdir(exist_ok=True); df=pd.read_csv(HERE/"titanic.csv") # never reloads seaborn
    xtr,xte,ytr,yte=train_test_split(df[FEATURES],df.survived,test_size=.2,random_state=42,stratify=df.survived)
    estimators={"logistic":LogisticRegression(max_iter=2000),"decision_tree":DecisionTreeClassifier(random_state=42),"random_forest":RandomForestClassifier(n_estimators=300,random_state=42)}
    fitted={}; rows=[]
    for name,model in estimators.items():
        fitted[name]=Pipeline([("preprocess",prep()),("model",model)]).fit(xtr,ytr); rows.append(score(name,fitted[name],xte,yte))
    comparison=pd.DataFrame(rows).round(3); comparison.to_csv(OUT/"classifier_comparison.csv",index=False)
    treepipe=fitted["decision_tree"]; fig,ax=plt.subplots(figsize=(20,10)); plot_tree(treepipe.named_steps["model"],feature_names=treepipe.named_steps["preprocess"].get_feature_names_out(),class_names=["not survived","survived"],filled=True,max_depth=3,ax=ax); fig.tight_layout(); fig.savefig(OUT/"decision_tree.png",dpi=150); plt.close(fig)
    variants={"baseline":Pipeline([("preprocess",prep()),("model",LogisticRegression(max_iter=2000))]),"class_weight_balanced":Pipeline([("preprocess",prep()),("model",LogisticRegression(max_iter=2000,class_weight="balanced"))]),"smote":ImbPipeline([("preprocess",prep()),("smote",SMOTE(random_state=42)),("model",LogisticRegression(max_iter=2000))])}
    imbalance=[]
    for name,pipe in variants.items():
        pipe.fit(xtr,ytr); p=pipe.predict(xte); imbalance.append({"variant":name,"precision":precision_score(yte,p),"recall":recall_score(yte,p),"f1":f1_score(yte,p)})
    imbalance=pd.DataFrame(imbalance).round(3); imbalance.to_csv(OUT/"imbalance_comparison.csv",index=False)
    search=GridSearchCV(Pipeline([("preprocess",prep()),("model",RandomForestClassifier(oob_score=True,random_state=42))]),{"model__n_estimators":[200,400],"model__max_depth":[None,6,12],"model__max_features":["sqrt","log2"]},cv=5,scoring="f1",n_jobs=-1).fit(xtr,ytr)
    best=search.best_estimator_; joblib.dump(best,OUT/"best_titanic_pipeline.joblib"); reload_check=joblib.load(OUT/"best_titanic_pipeline.joblib").predict(xte.iloc[:3]).tolist()
    rfeatures=["pclass","sex","age","sibsp","parch","embarked"]; rxtr,rxte,rytr,ryte=train_test_split(df[rfeatures],df.fare,test_size=.2,random_state=42)
    rprep=ColumnTransformer([("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),["pclass","age","sibsp","parch"]),("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore"))]),["sex","embarked"])])
    reg=Pipeline([("preprocess",rprep),("model",LinearRegression())]).fit(rxtr,rytr); rp=reg.predict(rxte); r2=r2_score(ryte,rp); n=len(ryte); k=reg.named_steps["preprocess"].transform(rxte).shape[1]
    rmetrics={"MAE":mean_absolute_error(ryte,rp),"RMSE":mean_squared_error(ryte,rp)**.5,"R2":r2,"Adjusted_R2":1-(1-r2)*(n-1)/(n-k-1)}; pd.DataFrame([rmetrics]).round(3).to_csv(OUT/"regression_metrics.csv",index=False)
    fig,ax=plt.subplots(); ax.scatter(rp,ryte-rp,alpha=.6); ax.axhline(0,color="red"); ax.set(xlabel="Predicted fare",ylabel="Residual"); fig.tight_layout(); fig.savefig(OUT/"regression_residuals.png",dpi=150); plt.close(fig)
    report=f'''# Modeling report

Class balance: {df.survived.value_counts(normalize=True).round(3).to_dict()}. Stratification preserves this balance in train and test. The split precedes all preprocessing; each imputer, encoder, and scaler is inside a pipeline fitted only on training data.

## Classification metrics
{comparison.to_markdown(index=False)}

Each model has a saved confusion matrix and ROC/AUC plot. The decision-tree image uses labelled transformed feature names and class names.

## Imbalance comparison
{imbalance.to_markdown(index=False)}
SMOTE is inside `ImbPipeline`, so oversampling occurs on training data only. Choose the variant with the strongest F1 while weighing its precision/recall trade-off.

GridSearchCV best parameters: `{search.best_params_}`; OOB score: `{best.named_steps['model'].oob_score_:.3f}`.

## Regression metrics (not comparable to classification metrics)
{pd.DataFrame([rmetrics]).round(3).to_markdown(index=False)}
Inspect `regression_residuals.png`: a fan-shaped/non-random residual spread is evidence of heteroscedasticity; this fare target's skew makes that a plausible conclusion.

## Recommendation
Deploy the classifier that leads on held-out F1 and AUC in the classification table, not accuracy alone because survival is imbalanced. A tuned random forest is preferred if it has that lead because it models nonlinear feature interactions. The saved `best_titanic_pipeline.joblib` contains preprocessing plus estimator and reloads on raw rows. Reload predictions: `{reload_check}`.
'''
    (OUT/"MODELING_REPORT.md").write_text(report,encoding="utf-8")
if __name__=="__main__": main()
