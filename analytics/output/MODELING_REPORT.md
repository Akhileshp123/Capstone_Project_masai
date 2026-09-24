# Modeling report

Class balance: {0: 0.616, 1: 0.384}. Stratification preserves this balance in train and test. The split precedes all preprocessing; each imputer, encoder, and scaler is inside a pipeline fitted only on training data.

## Classification metrics
| model         |   accuracy |   precision |   recall |    f1 |   auc |
|:--------------|-----------:|------------:|---------:|------:|------:|
| logistic      |      0.804 |       0.793 |    0.667 | 0.724 | 0.844 |
| decision_tree |      0.816 |       0.79  |    0.71  | 0.748 | 0.79  |
| random_forest |      0.81  |       0.797 |    0.681 | 0.734 | 0.829 |

Each model has a saved confusion matrix and ROC/AUC plot. The decision-tree image uses labelled transformed feature names and class names.

## Imbalance comparison
| variant               |   precision |   recall |    f1 |
|:----------------------|------------:|---------:|------:|
| baseline              |       0.793 |    0.667 | 0.724 |
| class_weight_balanced |       0.73  |    0.783 | 0.755 |
| smote                 |       0.74  |    0.783 | 0.761 |
SMOTE is inside `ImbPipeline`, so oversampling occurs on training data only. Choose the variant with the strongest F1 while weighing its precision/recall trade-off.

GridSearchCV best parameters: `{'model__max_depth': 6, 'model__max_features': 'sqrt', 'model__n_estimators': 200}`; OOB score: `0.822`.

## Regression metrics (not comparable to classification metrics)
|    MAE |   RMSE |   R2 |   Adjusted_R2 |
|-------:|-------:|-----:|--------------:|
| 20.809 | 30.473 |  0.4 |         0.368 |
Inspect `regression_residuals.png`: a fan-shaped/non-random residual spread is evidence of heteroscedasticity; this fare target's skew makes that a plausible conclusion.

## Recommendation
Deploy the classifier that leads on held-out F1 and AUC in the classification table, not accuracy alone because survival is imbalanced. A tuned random forest is preferred if it has that lead because it models nonlinear feature interactions. The saved `best_titanic_pipeline.joblib` contains preprocessing plus estimator and reloads on raw rows. Reload predictions: `[0, 0, 0]`.
