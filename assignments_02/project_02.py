import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# The student_performance_math.csv file uses a semicolon (;) as the separator, so pd.read_csv() needs sep=";" instead of the default comma separator.

# Task 1: Load and Explore
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")

print("Shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

plt.figure()
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.close()

# The table has loaded;
# Dimensions: 395 x 18;
# The columns sex, schoolsup, internet, higher, and activities are currently text-based;
# G3 is numeric;
# The histogram of the final grade (G3) has been saved to 'outputs'.

# ------------------------------------------------------------------------------
# Task 2: Preprocess the Data
print("\nShape before filtering G3 == 0:")
print(df.shape)

df_clean = df[df["G3"] != 0].copy()                                   # only the rows where G3 is not equal to zero

print("\nShape after filtering G3 == 0:")
print(df_clean.shape)

# Rows with G3 = 0 represent students who missed the final exam, not students who truly earned a score of zero.
# Keeping these rows would distort the model because the target would mix real academic performance with exam absence.

yes_no_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_cols:
    df_clean[col] = df_clean[col].map({"yes": 1, "no": 0})           # convert text-based binary columns into a numerical format

df_clean["sex"] = df_clean["sex"].map({"F": 0, "M": 1})              # convert text-based binary columns into a numerical format

corr_original, _ = stats.pearsonr(df["absences"], df["G3"])          # calculate the correlation between absences and G3.
corr_filtered, _ = stats.pearsonr(df_clean["absences"], df_clean["G3"])

print("\nPearson correlation between absences and G3 (original data):")
print(corr_original)

print("\nPearson correlation between absences and G3 (filtered data):")
print(corr_filtered)

# Filtering changes the result because students with G3 = 0 often had many absences and also missed the final exam. In the original data, 
# those rows mix exam absence with academic performance, which changes the relationship between absences and G3.
# Before filtering: 0.0342, that is almost no correlation. After filtering: -0.2131, the relationship becomes more meaningful in relation to actual academic performance.

# --------------------------------------------------------------------
# Task 3: Exploratory Data Analysis
numeric_features = [
    "age",
    "Medu",
    "Fedu",
    "traveltime",
    "studytime",
    "failures",
    "absences",
    "freetime",
    "goout",
    "Walc",
    "schoolsup",
    "internet",
    "higher",
    "activities",
    "sex",
]

correlations = []
for col in numeric_features:
    corr, _ = stats.pearsonr(df_clean[col], df_clean["G3"])
    correlations.append((col, corr))

correlations_sorted = sorted(correlations, key=lambda x: x[1])

print("\nCorrelations with G3 (sorted):")
for feature, corr in correlations_sorted:
    print(f"{feature:12s}: {corr:+.3f}")

'''
The strongest negative correlations with G3 are found in:
    failures: -0.294
    schoolsup: -0.238
    absences: -0.213
This means:
a higher number of past failures is associated with a lower G3 score;
a higher number of absences is also associated with a lower G3 score;
the negative correlation for 'schoolsup' does not mean that such support is "harmful" 
rather, the opposite is true: support is more frequently received by students who are already experiencing difficulties.

The strongest positive correlations are found in:
    Medu: +0.190
    Fedu: +0.159
    studytime: +0.127
This appears entirely logical:
more study time is associated with better final grades;
parents' education level is also associated with a higher G3 score.

It is somewhat surprising that:
    'activities' is almost neutral,
    'freetime' is almost neutral,
while 'sex' and 'internet' show only a weak positive correlation.    
'''

plt.figure()
plt.scatter(df_clean["failures"], df_clean["G3"])
plt.title("G3 vs Failures")
plt.xlabel("Failures")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/g3_vs_failures.png")
plt.close()

# This plot shows a negative relationship: students with more past failures tend to have lower final grades.

plt.figure()
plt.scatter(df_clean["studytime"], df_clean["G3"])
plt.title("G3 vs Study Time")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("assignments_02/outputs/g3_vs_studytime.png")
plt.close()

# This plot suggests a positive relationship: students who study more tend to have somewhat higher final grades,
# although the pattern is not extremely strong.

# -----------------------------------------------------------------
# Task 4: Baseline Model

X_baseline = df_clean[["failures"]].values
y = df_clean["G3"].values

X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
    X_baseline,
    y,
    test_size=0.2,
    random_state=42
)

baseline_model = LinearRegression()
baseline_model.fit(X_train_base, y_train_base)

y_pred_base = baseline_model.predict(X_test_base)

rmse_base = np.sqrt(np.mean((y_pred_base - y_test_base) ** 2))
r2_base = baseline_model.score(X_test_base, y_test_base)

print("\nBaseline model results:")
print(f"Slope: {baseline_model.coef_[0]}")
print(f"RMSE: {rmse_base}")
print(f"R^2: {r2_base}")

# The slope of -1.4275 means that each additional past failure is associated with about a 1.43-point decrease in predicted G3.
# The RMSE of 2.9617 means the model is typically off by about 2.96 grade points on a 0-20 scale, which is a fairly large prediction error.
# The test R^2 of 0.0895 means this baseline model explains only a small fraction of the variation in final grades.

# ----------------------------------------------------------------------------------
# Task 5: Build the Full Model
feature_cols = [
    "failures",
    "Medu",
    "Fedu",
    "studytime",
    "higher",
    "schoolsup",
    "internet",
    "sex",
    "freetime",
    "activities",
    "traveltime",
]

X = df_clean[feature_cols].values
y = df_clean["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)

y_pred = model_full.predict(X_test)

rmse_full = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2_train_full = model_full.score(X_train, y_train)
r2_test_full = model_full.score(X_test, y_test)

print("\nFull model results:")
print(f"Train R^2: {r2_train_full}")
print(f"Test R^2: {r2_test_full}")
print(f"RMSE: {rmse_full}")

print("\nFeature coefficients:")
for name, coef in zip(feature_cols, model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

coef_pairs = list(zip(feature_cols, model_full.coef_))
sorted_coefs = sorted(coef_pairs, key=lambda x: x[1])

print("\nLargest negative coefficients:")
for name, coef in sorted_coefs[:2]:
    print(f"{name:12s}: {coef:+.3f}")

print("\nLargest positive coefficients:")
for name, coef in sorted_coefs[-2:]:
    print(f"{name:12s}: {coef:+.3f}")

# The full model improves on the baseline: test R^2 increases from 0.0895 to 0.1539, so adding more features helps, but only modestly.
# The train R^2 is 0.1749 and the test R^2 is 0.1539, which are fairly close.
# That suggests the model is not strongly overfitting, but it also shows that the available features still do not explain most of the variation in G3.

# One surprising result is the large negative coefficient for schoolsup (-2.062).
# This does not necessarily mean school support lowers grades. A more likely explanation is that students receiving extra support were already struggling,
# so the variable is capturing prior academic risk rather than the effect of support itself.


# ------------------------------------------------------------------------
# Task 6: Evaluate and Summarize
plt.figure()
plt.scatter(y_pred, y_test)

min_value = min(y_pred.min(), y_test.min())
max_value = max(y_pred.max(), y_test.max())

plt.plot([min_value, max_value], [min_value, max_value], color="red")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("assignments_02/outputs/predicted_vs_actual_project02.png")
plt.close()

# A point above the diagonal means the actual grade is higher than the prediction, so the model underestimated the student's final grade.
# A point below the diagonal means the actual grade is lower than the prediction, so the model overestimated the student's final grade.
# The points are spread around the diagonal across the full grade range, but the predictions are compressed toward the middle, which suggests the model struggles
# to fully capture the highest and lowest outcomes.

print("\nSummary:")
print(f"Filtered dataset size: {df_clean.shape}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Best model RMSE: {rmse_full}")
print(f"Best model R^2: {r2_test_full}")

# On a 0-20 grade scale, an RMSE of 2.8550 means the model is typically off by almost 3 grade points, which is a noticeable prediction error in practice.
# The test R^2 of 0.1539 means the model explains only a limited share of the variation in final grades, even though it performs better than the baseline.

# The largest positive coefficients are internet and higher, which are associated with higher predicted G3 in this model.
# The largest negative coefficients are schoolsup and failures, which are associated with lower predicted G3.

# One surprising result is the strong negative coefficient for schoolsup.
# A likely explanation is that students receiving extra school support were already more academically at risk, so the variable reflects existing difficulty
# more than the effect of support itself.

# If deploying this model in production, I would keep features with clearer signal and practical meaning, such as failures, studytime, higher, internet, sex,
# Medu, and Fedu. I would consider dropping features with very small coefficients like activities and freetime unless there were domain reasons to keep them.

# Neglected Feature: The Power of G1
feature_cols_with_g1 = feature_cols + ["G1"]

X_g1 = df_clean[feature_cols_with_g1].values
y_g1 = df_clean["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(
    X_g1,
    y_g1,
    test_size=0.2,
    random_state=42
)

model_with_g1 = LinearRegression()
model_with_g1.fit(X_train_g1, y_train_g1)

r2_test_with_g1 = model_with_g1.score(X_test_g1, y_test_g1)

print("\nModel with G1 included:")
print(f"Test R^2 with G1: {r2_test_with_g1}")

# Adding G1 increases the test R^2 from 0.1539 to 0.7491, which is a very large jump.
# This does not mean G1 causes G3; it means G1 is a very strong predictor because it is an earlier grade from the same course and reflects prior performance in the same subject.
# This model could be useful for identifying students at risk after the first grading period, but it is less useful for very early intervention before G1 is available.
# If educators want to intervene earlier, they need features that are available before G1 exists, such as attendance, study habits, prior failures, and background factors.