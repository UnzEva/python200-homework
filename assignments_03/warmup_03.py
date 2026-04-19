import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Q1. Preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Q2. Preprocessing
# -----------------------------------------------------------------------------------
scaler = StandardScaler()

# Calculate the mean and standard deviation, and scales the train set
X_train_scaled = scaler.fit_transform(X_train)
# Apply the same mean and std to the test set
X_test_scaled = scaler.transform(X_test)

print("Column means in X_train_scaled:")
# Calculate the mean for each feature
print(X_train_scaled.mean(axis=0))

# We fit the scaler on X_train only to avoid leaking information from the test set into the preprocessing step.

# Q1. KNN
# -----------------------------------------------------------------------------------
# Create a KNeighborsClassifier with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the classifier on unscaled data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Print the accuracy and the full classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print("Classification report:")
print(classification_report(y_test, y_pred))

# The model correctly classified all 30 objects in the test set, made no errors in any of the classes.
# This is possible for this dataset because it is small, clean, and the classes are well-separable.

# Q2. KNN
# -----------------------------------------------------------------------------------

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

y_pred_scaled = knn_scaled.predict(X_test_scaled)

print(f"Scaled accuracy: {accuracy_score(y_test, y_pred_scaled)}")

# Scaling slightly worsened the result:
#         without scaling: 1.0
#         with scaling: 0.9333
# For this particular dataset split, the original features already work very well for KNN, so scaling changes neighbor distances without improving the classification.

# Q3. KNN
# -----------------------------------------------------------------------------------
knn_cv = KNeighborsClassifier(n_neighbors=5)

# Using cross-validation, train the model multiple times and calculate the accuracy on each fold.
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

print("Cross-validation scores:")
print(cv_scores)
print(f"Mean CV score: {cv_scores.mean()}")
print(f"Standard deviation: {cv_scores.std()}")

# This result is more trustworthy than a single train/test split because cross-validation evaluates the model across multiple folds instead of
# relying on just one particular split of the data.

# Q4. KNN
# -----------------------------------------------------------------------------------
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

# Iterate through different options for the number of neighbors
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores_k = cross_val_score(knn_k, X_train, y_train, cv=5)
    print(f"k={k}, mean CV score={scores_k.mean()}")

# I would choose k=7 because it ties for the best mean cross-validation score
# and is slightly less sensitive to noise than a smaller value like k=5

# Q1. Classifier Evaluation
# -----------------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("assignments_03/outputs/knn_confusion_matrix.png")
plt.close()

# The model does not confuse any pair of species here, because all predictions are correct and the confusion matrix is fully diagonal.

# Q1. Decision Trees
# -----------------------------------------------------------------------------------
# Create a model with a tree depth=3
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

y_pred_tree = tree.predict(X_test)

print(f"Decision Tree accuracy: {accuracy_score(y_test, y_pred_tree)}")
print("Decision Tree classification report:")
print(classification_report(y_test, y_pred_tree))

# The Decision Tree performs slightly worse than KNN on this split, because its accuracy is 0.9667 while KNN reached 1.0
#  Scaled vs. unscaled data would usually make little or no difference for Decision Trees because they do not depend on distance calculations.

# Q1. Logistic Regression and Regularization
# -----------------------------------------------------------------------------------
c_values = [0.01, 1.0, 100]

# Train three Logistic Regression models with different values ​​for the parameter C, 
# which controls regularization (small C means stronger regularization; large C means weaker regularization).
for c_value in c_values:
    log_model = OneVsRestClassifier(                                 # OneVsRestClassifier transforms a multiclass problem into several binary ones.
        LogisticRegression(
            C=c_value,
            max_iter=1000,
            solver="liblinear"
        )
    )
    log_model.fit(X_train_scaled, y_train)

    coefficient_size = np.abs(log_model.estimators_[0].coef_).sum()   # After wrapping, `log_model.estimators` no longer contains a single model, but rather several separate Logistic Regression models.
    for estimator in log_model.estimators_[1:]:
        coefficient_size += np.abs(estimator.coef_).sum()             # coefficients can be positive or negative.

    print(f"C={c_value}, total coefficient size={coefficient_size}")

# As C increases, the total coefficient magnitude increases.
# This shows that weaker regularization allows larger coefficients, while stronger regularization shrinks them toward zero.

# --- PCA ---

digits = load_digits()
X_digits = digits.data
y_digits = digits.target
images = digits.images

# Q1. PCA
#-------------------------------------------------------------------------------------------------
print(f"X_digits shape: {X_digits.shape}")
print(f"images shape: {images.shape}")

fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for digit in range(10):
    first_index = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[first_index], cmap="gray_r")         # Displays the image in an inverted grayscale palette
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("assignments_03/outputs/sample_digits.png")
plt.close()

# Q2. PCA
#-------------------------------------------------------------------------------------------------
# PCA first computes all components
pca = PCA()

pca.fit(X_digits)

# Convert each image into coordinates based on PCA components
scores = pca.transform(X_digits)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap="tab10", s=10)
plt.colorbar(scatter, label="Digit")
plt.title("PCA 2D Projection of Digits")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("assignments_03/outputs/pca_2d_projection.png")
plt.close()

# Same-digit images tend to cluster together to some extent in this 2D space, although the clusters are not perfectly separated.

# Q3. PCA
#-------------------------------------------------------------------------------------------------

cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)         # number the components starting from 1, not 0
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("assignments_03/outputs/pca_variance_explained.png")
plt.close()

components_for_80 = np.argmax(cumulative_variance >= 0.80) + 1
print(f"Components needed to explain about 80% of the variance: {components_for_80}")

#

# Q4. PCA
#-------------------------------------------------------------------------------------------------
# 
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)


n_values = [2, 5, 15, 40]

fig, axes = plt.subplots(len(n_values) + 1, 5, figsize=(10, 10))

# Original row
for col in range(5):
    axes[0, col].imshow(images[col], cmap="gray_r")
    if col == 0:
        axes[0, col].set_ylabel("Original")
    axes[0, col].set_title(f"Digit {y_digits[col]}")
    axes[0, col].axis("off")

# Reconstructed rows
for row, n in enumerate(n_values, start=1):
    for col in range(5):
        reconstructed = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(reconstructed, cmap="gray_r")
        if col == 0:
            axes[row, col].set_ylabel(f"n={n}")
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("assignments_03/outputs/pca_reconstructions.png")
plt.close()

# The digits usually become clearly recognizable by around n=15.
# That generally matches the point where the variance curve has already captured a large share of the total information and starts leveling off.