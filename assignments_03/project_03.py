import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# The Spambase dataset file has no header row, so we assign column names manually when loading it with pandas.

feature_names = [
    "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d",
    "word_freq_our", "word_freq_over", "word_freq_remove", "word_freq_internet",
    "word_freq_order", "word_freq_mail", "word_freq_receive", "word_freq_will",
    "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
    "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money",
    "word_freq_hp", "word_freq_hpl", "word_freq_george", "word_freq_650",
    "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857",
    "word_freq_data", "word_freq_415", "word_freq_85", "word_freq_technology",
    "word_freq_1999", "word_freq_parts", "word_freq_pm", "word_freq_direct",
    "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference",
    "char_freq_;", "char_freq_(", "char_freq_[", "char_freq_!", "char_freq_$",
    "char_freq_#", "capital_run_length_average", "capital_run_length_longest",
    "capital_run_length_total", "spam_label"
]

# --- Task 1: Load and Explore ---
#-------------------------------------------------------------------------------------------------
df = pd.read_csv(
    "assignments_03/spambase.data",
    header=None,
    names=feature_names
)

print("Dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

print("\nClass counts:")
print(df["spam_label"].value_counts())

print("\nClass proportions:")
print(df["spam_label"].value_counts(normalize=True))

print("\nClass balance interpretation:")
print("Ham is the majority class, so accuracy alone can be misleading.")
print("A model that predicts ham too often may still look good on raw accuracy.")

# The classes are not perfectly balanced, so raw accuracy should be interpreted carefully.
# A model can look strong on accuracy alone if it mainly predicts the majority class.
# For spam filtering, it is also important to examine the types of errors the model makes.

# Boxplot 1: word_freq_free
# Spam emails are expected to show higher values for word_freq_free, while many ham emails likely remain near zero because most emails do not use this word.
plt.figure()
df.boxplot(column="word_freq_free", by="spam_label")
plt.title("word_freq_free by Spam Label")
plt.suptitle("")
plt.xlabel("Spam Label (0 = ham, 1 = spam)")
plt.ylabel("word_freq_free")
plt.savefig("assignments_03/outputs/word_freq_free_boxplot.png")
plt.close()

# Boxplot 2: char_freq_!
# Spam emails may use exclamation marks more aggressively than ham emails, so the spam distribution may sit higher and show more extreme outliers.
plt.figure()
df.boxplot(column="char_freq_!", by="spam_label")
plt.title("char_freq_! by Spam Label")
plt.suptitle("")
plt.xlabel("Spam Label (0 = ham, 1 = spam)")
plt.ylabel("char_freq_!")
plt.savefig("assignments_03/outputs/char_freq_exclamation_boxplot.png")
plt.close()

# Boxplot 3: capital_run_length_total
# Spam emails may show much larger capital letter runs, which can reflect attention-grabbing formatting such as ALL CAPS words or phrases.
plt.figure()
df.boxplot(column="capital_run_length_total", by="spam_label")
plt.title("capital_run_length_total by Spam Label")
plt.suptitle("")
plt.xlabel("Spam Label (0 = ham, 1 = spam)")
plt.ylabel("capital_run_length_total")
plt.savefig("assignments_03/outputs/capital_run_length_total_boxplot.png")
plt.close()

# Many word-frequency features are heavily skewed toward zero because most emails do not contain many of the tracked words at all. That means the dataset is sparse
# in an intuitive sense: lots of feature values are exactly zero or very close to zero.
# The feature scales also vary dramatically. Some word and character frequencies are small percentages or fractions, while features such as capital_run_length_total can be much larger. 
# This matters for models like KNN and logistic regression, because those models are sensitive to feature scale. Larger-scale features can dominate
# distances or coefficients unless the data is standardized first.

# --- Task 2: Prepare Your Data ---
#----------------------------------------------------------------------------------------------
X = df.drop(columns="spam_label")                               # We take all features except the target
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(            # We use stratify=y so that the spam/ham class balance stays similar in both the training set and the test set.
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("\nTrain/Test shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")



scaler = StandardScaler()                                          # We fit the scaler on X_train only to avoid leaking information from the test set into preprocessing.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train_scaled)

cumulative_variance = np.cumsum(pca.explained_variance_ratio_) 
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1       # We find the first component where the cumulative explained variance reaches at least 0.90.

print(f"\nNumber of PCA components needed for 90% variance: {n_components_90}")

plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
plt.axhline(0.90, color="red", linestyle="--")
plt.axvline(n_components_90, color="red", linestyle="--")
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("assignments_03/outputs/pca_cumulative_variance.png")
plt.close()

X_train_pca = pca.transform(X_train_scaled)[:, :n_components_90]
X_test_pca = pca.transform(X_test_scaled)[:, :n_components_90]

print("\nScaled and PCA-reduced shapes:")
print(f"X_train_scaled: {X_train_scaled.shape}")
print(f"X_test_scaled: {X_test_scaled.shape}")
print(f"X_train_pca: {X_train_pca.shape}")
print(f"X_test_pca: {X_test_pca.shape}")

# PCA is fit on the scaled training data only. This prevents test-set information from influencing the learned components and keeps evaluation honest.
# Scaling is especially important here because the feature magnitudes vary a lot: some are small frequencies, while others are much larger run-length counts.

# --- Task 3: A Classifier Comparison ---
# ------------------------------------------------------------------------------------------
# KNN on unscaled data
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)

y_pred_knn_unscaled = knn_unscaled.predict(X_test)

knn_unscaled_accuracy = accuracy_score(y_test, y_pred_knn_unscaled)

print("\nKNN (unscaled data):")
print(f"Accuracy: {knn_unscaled_accuracy}")
print("Classification report:")
print(classification_report(y_test, y_pred_knn_unscaled))

# KNN uses distance between samples, so large-scale features can dominate smaller-scale ones. 
# This unscaled result gives us a baseline to compare against scaled and PCA-reduced versions.

#----------------------------------------------------------------------------------------------------
# KNN on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_knn_scaled = knn_scaled.predict(X_test_scaled)

knn_scaled_accuracy = accuracy_score(y_test, y_pred_knn_scaled)

print("\nKNN (scaled data):")
print(f"Accuracy: {knn_scaled_accuracy}")
print("Classification report:")
print(classification_report(y_test, y_pred_knn_scaled))

# KNN is distance-based, so scaling often helps by preventing large-scale features from dominating the distance calculation.
# This result should be compared directly with the unscaled KNN model.

# ---------------------------------------------------------------------------------
# KNN on PCA-reduced data
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)

y_pred_knn_pca = knn_pca.predict(X_test_pca)

knn_pca_accuracy = accuracy_score(y_test, y_pred_knn_pca)

print("\nKNN (PCA-reduced data):")
print(f"Accuracy: {knn_pca_accuracy}")
print("Classification report:")
print(classification_report(y_test, y_pred_knn_pca))

# PCA may help KNN by compressing the data into a smaller set of informative directions, but it can also remove useful detail. 
# We compare this result directly with scaled KNN to see which version works better here.

#---------------------------------------------------------------------------------
# Decision Tree depth comparison
depth_values = [3, 5, 10, None]

print("\nDecision Tree depth comparison:")
for depth in depth_values:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree_model.fit(X_train, y_train)

    train_accuracy = tree_model.score(X_train, y_train)
    test_accuracy = tree_model.score(X_test, y_test)

    print(f"max_depth={depth}, train accuracy={train_accuracy}, test accuracy={test_accuracy}")

# As tree depth increases, training accuracy usually rises.
# If test accuracy does not improve in the same way, that is evidence of overfitting.

#-----------------------------------------------------------------------------------------
# Final Decision Tree model
tree_final = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_final.fit(X_train, y_train)

y_pred_tree = tree_final.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree (chosen depth=10):")
print(f"Accuracy: {tree_accuracy}")
print("Classification report:")
print(classification_report(y_test, y_pred_tree))

# I would choose max_depth=10 for production.
# The unlimited tree reaches almost perfect training accuracy, which shows that it is close to memorizing the training data. 
# However, its test accuracy improves only slightly compared with max_depth=10.
#
# That pattern reflects the bias-variance tradeoff: deeper trees reduce bias on the
# training set, but they also increase variance and become more sensitive to noise.
# A depth of 10 keeps nearly the same test performance while using a simpler model
# that is less likely to memorize spurious patterns in the training data.

#----------------------------------------------------------------------------------------------
# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("\nRandom Forest:")
print(f"Accuracy: {rf_accuracy}")
print("Classification report:")
print(classification_report(y_test, y_pred_rf))

# Random Forest averages predictions across many trees, which usually makes it more stable and less prone to overfitting than a single Decision Tree.

# ---------------------------------------------------------------------------------------------
# Logistic Regression on scaled data
logreg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
logreg_scaled.fit(X_train_scaled, y_train)

y_pred_logreg_scaled = logreg_scaled.predict(X_test_scaled)

logreg_scaled_accuracy = accuracy_score(y_test, y_pred_logreg_scaled)

print("\nLogistic Regression (scaled data):")
print(f"Accuracy: {logreg_scaled_accuracy}")
print("Classification report:")
print(classification_report(y_test, y_pred_logreg_scaled))

# Logistic regression is sensitive to feature scale, so it is appropriate to train it on standardized data rather than the raw feature values.

# -----------------------------------------------------------------------------------------------
# Logistic Regression on PCA-reduced data
logreg_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
logreg_pca.fit(X_train_pca, y_train)

y_pred_logreg_pca = logreg_pca.predict(X_test_pca)

logreg_pca_accuracy = accuracy_score(y_test, y_pred_logreg_pca)

print("\nLogistic Regression (PCA-reduced data):")
print(f"Accuracy: {logreg_pca_accuracy}")
print("Classification report:")
print(classification_report(y_test, y_pred_logreg_pca))

# PCA can reduce dimensionality and remove some noise, but it may also discard useful information. 
# This comparison shows whether PCA helps logistic regression on this dataset or whether the full scaled feature set works better.

# ------------------------------------------------------------------------------------------------
# Feature importances for Decision Tree and Random Forest
tree_importances = pd.Series(tree_final.feature_importances_, index=X.columns)
rf_importances = pd.Series(rf_model.feature_importances_, index=X.columns)

tree_top10 = tree_importances.sort_values(ascending=False).head(10)
rf_top10 = rf_importances.sort_values(ascending=False).head(10)

print("\nTop 10 Decision Tree feature importances:")
for feature, importance in tree_top10.items():
    print(f"{feature:30s} {importance:.4f}")

print("\nTop 10 Random Forest feature importances:")
for feature, importance in rf_top10.items():
    print(f"{feature:30s} {importance:.4f}")

plt.figure(figsize=(10, 6))
rf_top10.sort_values().plot(kind="barh")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("assignments_03/outputs/feature_importances.png")
plt.close()

# The two models agree on several important spam-related features, especially char_freq_!, char_freq_$, word_freq_remove, and word_freq_free.
#
# These features make intuitive sense for spam detection. 
# Spam messages often use attention-grabbing punctuation such as exclamation marks and dollar signs, and
# they frequently contain words like "free" or "remove" that appear in promotional or manipulative email.
#
# Random Forest spreads importance across several related features, including the capital-run-length statistics, 
# which suggests that spam detection benefits from combining many weak-to-moderate signals rather than relying on a single rule.

# -----------------------------------------------------------------------------------------------
# Task 3 summary
# Random Forest performs best on this dataset because spam detection here depends on many nonlinear patterns and feature interactions. 
# The dataset mixes word frequencies, character frequencies, and capitalization statistics, and a forest
# can capture complex threshold-based rules without requiring feature scaling.

# KNN performs very poorly on unscaled data because it relies directly on distances between samples. 
# In Spambase, some features are tiny frequencies while others, such as capital-run-length features, can be much larger. 
# Without scaling, the larger-magnitude features dominate the distance calculation and drown out useful signal from smaller-scale features.

# Scaling helps KNN dramatically because it puts all features on a comparable scale.
# That makes the distance calculation more meaningful and allows word-frequency, character-frequency, and capitalization features to contribute more fairly.

# PCA does not help much here and can slightly hurt performance because PCA preserves directions of high variance, 
# not necessarily directions with the most predictive signal for spam classification. 
# Some lower-variance features may still be highly useful for separating spam from ham. 
# PCA is also a linear transformation, so it cannot preserve every nonlinear structure that models like KNN or forests may use.

# Logistic Regression still performs well because the dataset contains many signals that are already fairly informative in a linear way. 
# Features such as exclamation frequency, dollar-sign frequency, and words like "free" or "remove" can push the
# prediction toward spam even without modeling highly complex interactions.

# For a spam filter, accuracy is useful but not sufficient on its own. 
# False positives mean legitimate email is marked as spam, which can be very costly if an important message is hidden or missed. 
# False negatives mean spam gets through, which is annoying and potentially risky.

# In a real spam-filtering system, I would usually prioritize reducing false positives,
# which means pushing precision on the spam class higher. 
# One practical way to do that would be threshold tuning with predict_proba(), 
# so that the model only labels an email as spam when its predicted spam probability exceeds a stricter cutoff.

#-----------------------------------------------------------------------------------------------
# Confusion matrix for the best-performing model (Random Forest)
cm_best = confusion_matrix(y_test, y_pred_rf)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=["ham", "spam"])
disp.plot()
plt.title("Best Model Confusion Matrix")
plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")
plt.close()

tn, fp, fn, tp = cm_best.ravel()

print("\nBest model error breakdown:")
print(f"False positives (ham predicted as spam): {fp}")
print(f"False negatives (spam predicted as ham): {fn}")

# In this result, false negatives are more common than false positives.
# That means the model more often lets spam through than incorrectly blocks legitimate email. 
# Depending on the business setting, that may or may not be the right tradeoff.
#
# If avoiding false positives is the top priority, a production system could use
# predict_proba() and raise the threshold for labeling an email as spam. 
# That would usually increase spam precision, but it would also allow more spam messages through.

# --- Task 4: Cross-Validation ---
# ----------------------------------------------------------------------------------------------

cv_results = {}

# KNN on unscaled data
scores_knn_unscaled = cross_val_score(
    KNeighborsClassifier(n_neighbors=5),
    X_train,
    y_train,
    cv=5
)
cv_results["KNN unscaled"] = scores_knn_unscaled

# KNN on scaled data
knn_scaled_pipeline_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

scores_knn_scaled = cross_val_score(
    knn_scaled_pipeline_cv,
    X_train,
    y_train,
    cv=5
)
cv_results["KNN scaled"] = scores_knn_scaled

# KNN on PCA-reduced data
knn_pca_pipeline_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components_90)),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

scores_knn_pca = cross_val_score(
    knn_pca_pipeline_cv,
    X_train,
    y_train,
    cv=5
)
cv_results["KNN PCA"] = scores_knn_pca

# Decision Tree
scores_tree = cross_val_score(
    DecisionTreeClassifier(max_depth=10, random_state=42),
    X_train,
    y_train,
    cv=5
)
cv_results["Decision Tree"] = scores_tree

# Random Forest
scores_rf = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_train,
    y_train,
    cv=5
)
cv_results["Random Forest"] = scores_rf

# Logistic Regression on scaled data
logreg_scaled_pipeline_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

scores_logreg_scaled = cross_val_score(
    logreg_scaled_pipeline_cv,
    X_train,
    y_train,
    cv=5
)
cv_results["Logistic Regression scaled"] = scores_logreg_scaled

# Logistic Regression on PCA-reduced data
logreg_pca_pipeline_cv = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n_components_90)),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

scores_logreg_pca = cross_val_score(
    logreg_pca_pipeline_cv,
    X_train,
    y_train,
    cv=5
)
cv_results["Logistic Regression PCA"] = scores_logreg_pca

print("\nCross-validation results:")
for model_name, scores in cv_results.items():
    print(f"{model_name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

best_mean_model = max(cv_results.items(), key=lambda item: item[1].mean())
most_stable_model = min(cv_results.items(), key=lambda item: item[1].std())

print("\nCross-validation summary:")
print(f"Most accurate model by mean CV score: {best_mean_model[0]} ({best_mean_model[1].mean():.4f})")
print(f"Most stable model by CV std: {most_stable_model[0]} ({most_stable_model[1].std():.4f})")

# These cross-validation results are more trustworthy because scaling and PCA
# now happen inside pipelines. That means each fold fits preprocessing only on
# that fold's training split, which avoids leakage from the validation fold.

# --- Task 5: Building a Prediction Pipeline ---
# ----------------------------------------------------------------------------------------------------

# Best tree-based pipeline: Random Forest
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf_pipeline = rf_pipeline.predict(X_test)

print("\nRandom Forest pipeline:")
print("Classification report:")
print(classification_report(y_test, y_pred_rf_pipeline))


# Best non-tree-based pipeline: Logistic Regression with scaling
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver="liblinear"))
])

logreg_pipeline.fit(X_train, y_train)
y_pred_logreg_pipeline = logreg_pipeline.predict(X_test)

print("\nLogistic Regression pipeline:")
print("Classification report:")
print(classification_report(y_test, y_pred_logreg_pipeline))

# These two pipelines do not have exactly the same structure.
# The Random Forest pipeline does not need scaling because tree-based models
# split on feature thresholds and are not driven by distances or coefficient scale.
# The Logistic Regression pipeline includes a scaler because logistic regression
# is sensitive to feature magnitudes and works better when features are standardized.

# Packaging preprocessing and modeling into a pipeline is valuable because it makes
# the workflow easier to reuse, reduces the risk of forgetting a preprocessing step,
# and helps prevent train/test inconsistencies when the model is handed off or deployed.