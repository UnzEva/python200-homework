import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt


# --- scikit-learn API ---

# Q1 scikit-learn
years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)                      # `reshape` is needed because scikit-learn expects features in the form of a 2D array
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Create the model
model = LinearRegression()

# Fit the model to the training data
model.fit(years, salary)

# Make predictions for new values
prediction_4_years = model.predict([[4]])[0]                            # [0] is needed because predict() returns an array                      
prediction_8_years = model.predict([[8]])[0]

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Predicted salary for 4 years of experience: {prediction_4_years}")
print(f"Predicted salary for 8 years of experience: {prediction_8_years}")

# ---------------------------------------------------------------------------------------------------
# Q2 scikit-learn
x = np.array([10, 20, 30, 40, 50])

print(f"Original shape: {x.shape}")

x_2d = x.reshape(-1, 1)

print(f"Reshaped shape: {x_2d.shape}")

# scikit-learn expects X to be 2D because it treats data as rows and columns:
# each row is one sample and each column is one feature.

# ---------------------------------------------------------------------------------------------------
# Q3 scikit-learn
X_clusters, _ = make_blobs(
    n_samples=120,
    centers=3,
    cluster_std=0.8,
    random_state=7
)

# Create the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(X_clusters)

# Predict cluster labels for all points
labels = kmeans.predict(X_clusters)

# Print cluster centers and cluster sizes
print("Cluster centers:")
print(kmeans.cluster_centers_)

print("Points in each cluster:")
print(np.bincount(labels))

# Create and save the scatter plot
plt.figure()
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200,
    c="black"
)
plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("outputs/kmeans_clusters.png")
# Close the current plot after saving
plt.close()

# # --- Linear Regression ---

# Q1 Linear Regression
np.random.seed(42)                                                 # needs to be generated identically every time
num_patients = 100
age = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)


# Linear Regression Q1
plt.figure()
plt.scatter(age, cost, c=smoker, cmap="coolwarm")                  # 'cmap' sets the color palette
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.close()

# The plot shows two visible groups.
# Smokers tend to have much higher medical costs than non-smokers.
# This suggests that smoker status is an important predictor of cost.

# -------------------------------------------------------------------------------------------------------
# Q2 Linear Regression 
X = age.reshape(-1, 1)                                        # converts the 'age' column into a 2D array.
y = cost                                                      # target

X_train, X_test, y_train, y_test = train_test_split(          # 80/20 split: train/test           
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# -------------------------------------------------------------------------------------------------------
# Q3 Linear Regression 
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)                                # train the model exclusively on the training data

y_pred = lr_model.predict(X_test)                             # generate predictions for the test data

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))               # calculate RMSE
r2_test = lr_model.score(X_test, y_test)                      # calculate R^2

print(f"Slope: {lr_model.coef_[0]}")
print(f"Intercept: {lr_model.intercept_}")
print(f"RMSE: {rmse}")
print(f"R^2 on the test set: {r2_test}")

# The slope shows how much predicted medical cost increases, on average, for each additional year of age in this simple linear regression model.
# A model based only on age performs poorly, as the R^2 value is around 0.07; 
# this is a very low value, indicating that age alone is insufficient to explain the cost.

# -------------------------------------------------------------------------------------------------------
# Q4 Linear Regression 
X_full = np.column_stack([age, smoker])                       # create X_full from two features

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full,
    cost,
    test_size=0.2,
    random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_full = model_full.score(X_test_full, y_test_full)

print(f"R^2 from Q3 (age only): {r2_test}")
print(f"R^2 from Q4 (age + smoker): {r2_full}")
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# The smoker coefficient shows how much higher predicted medical cost is, on average, for smokers compared with non-smokers, holding age constant.
# R^2 increased from 0.0695 to 0.7737, meaning that adding the 'smoker' variable significantly improved the model.

# -------------------------------------------------------------------------------------------------------
# Q5 Linear Regression 
y_pred_full = model_full.predict(X_test_full)                 # take the predictions specifically from the model in Q4.

plt.figure()
plt.scatter(y_pred_full, y_test_full)

min_value = min(y_pred_full.min(), y_test_full.min())
max_value = max(y_pred_full.max(), y_test_full.max())

plt.plot([min_value, max_value], [min_value, max_value], color="red")           # the ideal line of coincidence between predicted and actual values
plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# A point above the diagonal means the actual value is higher than the prediction, so the model underestimated the cost.
# A point below the diagonal means the actual value is lower than the prediction, so the model overestimated the cost.
