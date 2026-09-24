# EDA report

Original shape: (891, 15). Missing percentages: {'age': 19.87, 'embarked': 0.22, 'deck': 77.22, 'embark_town': 0.22}. `embarked` is 0.22% missing (<5%), so its rows are dropped. `age` is 19.87% (5–30%), so it is median-imputed. `deck` is 77.22% missing, so imputation is unreliable and the column is dropped.

Age IQR outliers: 65; fare IQR outliers: 114. Fare mean/median/mode are 32.10/14.45/8.05; mean > median > mode indicates a right-skewed distribution. `zscore_check.csv` shows age and fare have approximately mean 0 and standard deviation 1 after exploratory z-score standardization; modelling scales separately after splitting.

Survival rate by sex: {'female': 0.74, 'male': 0.189}. By pclass: {1: 0.626, 2: 0.473, 3: 0.242}. Boolean masking by sex and pclass: {'female_1': np.float64(0.967), 'female_2': np.float64(0.921), 'female_3': np.float64(0.5), 'male_1': np.float64(0.369), 'male_2': np.float64(0.157), 'male_3': np.float64(0.135)}. The heatmap contains exactly survived, pclass, age, sibsp, parch, fare; its strongest absolute off-diagonal pairs are [('pclass', 'fare', -0.548), ('sibsp', 'parch', 0.415)]. They show association, not causation.

## Four-chart story

1. `survival_by_sex.png`: the sex gap makes sex an important predictive candidate.
2. `survival_by_class.png`: first-class passengers survive more often, suggesting class-related advantage.
3. `age_by_survival.png`: the overlapping age distributions show age alone cannot decide survival.
4. `fare_age_survival.png`: survivors occur more often at higher fares, but overlap motivates multifeature modelling.
