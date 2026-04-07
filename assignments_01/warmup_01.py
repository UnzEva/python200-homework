import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns


# --- Pandas Review ---

# Pandas Q1
data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade": [85, 72, 90, 68, 95],
    "city": ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True],
}

df = pd.DataFrame(data)

print("Pandas Q1 - First three rows:")
print(df.head(3))

print("\nPandas Q1 - Shape:")
print(df.shape)

print("\nPandas Q1 - Data types:")
print(df.dtypes)


# Pandas Q2
filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]

print("\nPandas Q2 - Students who passed and have a grade above 80:")
print(filtered_df)


# Pandas Q3
df["grade_curved"] = df["grade"] + 5

print("\nPandas Q3 - DataFrame with grade_curved:")
print(df)


# Pandas Q4
df["name_upper"] = df["name"].str.upper()

print("\nPandas Q4 - name and name_upper columns:")
print(df[["name", "name_upper"]])


# Pandas Q5
mean_grade_by_city = df.groupby("city")["grade"].mean()

print("\nPandas Q5 - Mean grade by city:")
print(mean_grade_by_city)


# Pandas Q6
df["city"] = df["city"].replace("Austin", "Houston")

print("\nPandas Q6 - name and city columns after replacing Austin with Houston:")
print(df[["name", "city"]])


# Pandas Q7
sorted_df = df.sort_values(by="grade", ascending=False)

print("\nPandas Q7 - Top 3 rows sorted by grade descending:")
print(sorted_df.head(3))

# --- NumPy Review ---

# NumPy Q1
arr_1d = np.array([10, 20, 30, 40, 50])

print("\nNumPy Q1 - Array:")
print(arr_1d)
print("Shape:", arr_1d.shape)
print("Dtype:", arr_1d.dtype)
print("Ndim:", arr_1d.ndim)


# NumPy Q2
arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print("\nNumPy Q2 - 2D array:")
print(arr)
print("Shape:", arr.shape)
print("Size:", arr.size)


# NumPy Q3
top_left_block = arr[:2, :2]

print("\nNumPy Q3 - Top-left 2x2 block:")
print(top_left_block)


# NumPy Q4
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 5))

print("\nNumPy Q4 - 3x4 zeros array:")
print(zeros_array)

print("\nNumPy Q4 - 2x5 ones array:")
print(ones_array)


# NumPy Q5
arange_array = np.arange(0, 50, 5)

print("\nNumPy Q5 - np.arange(0, 50, 5):")
print(arange_array)
print("Shape:", arange_array.shape)
print("Mean:", np.mean(arange_array))
print("Sum:", np.sum(arange_array))
print("Standard deviation:", np.std(arange_array))


# NumPy Q6
random_array = np.random.normal(loc=0, scale=1, size=200)

print("\nNumPy Q6 - Random normal array stats:")
print("Mean:", np.mean(random_array))
print("Standard deviation:", np.std(random_array))

# --- Matplotlib Review ---

# Matplotlib Q1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

print("\nMatplotlib Q1 - Line plot created.")

plt.figure()
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Matplotlib Q2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

print("\nMatplotlib Q2 - Bar plot created.")

plt.figure()
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subject")
plt.ylabel("Score")
plt.show()


# Matplotlib Q3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

print("\nMatplotlib Q3 - Scatter plot created.")

plt.figure()
plt.scatter(x1, y1, label="Dataset 1", color="blue")
plt.scatter(x2, y2, label="Dataset 2", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# Matplotlib Q4
print("\nMatplotlib Q4 - Subplots created.")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(x, y)
axes[0].set_title("Squares")

axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")

plt.tight_layout()
plt.show()

# --- Descriptive Statistics Review ---

# Descriptive Stats Q1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print("\nDescriptive Stats Q1 - Mean:", np.mean(data))
print("Descriptive Stats Q1 - Median:", np.median(data))
print("Descriptive Stats Q1 - Variance:", np.var(data))
print("Descriptive Stats Q1 - Standard deviation:", np.std(data))


# Descriptive Stats Q2
scores = np.random.normal(65, 10, 500)

print("\nDescriptive Stats Q2 - Histogram created.")

plt.figure()
plt.hist(scores, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.show()


# Descriptive Stats Q3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

print("\nDescriptive Stats Q3 - Boxplot created.")

plt.figure()
plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Score")
plt.show()


# Descriptive Stats Q4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

print("\nDescriptive Stats Q4 - Distribution comparison boxplots created.")

plt.figure()
plt.boxplot([normal_data, skewed_data], labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()

# The exponential distribution is more skewed.
# The mean is appropriate for the normal distribution.
# The median is more appropriate for the exponential distribution
# because it is less affected by skew and extreme values.


# Descriptive Stats Q5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

mode_data1 = stats.mode(data1, keepdims=False)
mode_data2 = stats.mode(data2, keepdims=False)

print("\nDescriptive Stats Q5 - data1 mean:", np.mean(data1))
print("Descriptive Stats Q5 - data1 median:", np.median(data1))
print("Descriptive Stats Q5 - data1 mode:", mode_data1.mode)

print("\nDescriptive Stats Q5 - data2 mean:", np.mean(data2))
print("Descriptive Stats Q5 - data2 median:", np.median(data2))
print("Descriptive Stats Q5 - data2 mode:", mode_data2.mode)

# The mean and median are very different for data2 because the value 150
# is an outlier that pulls the mean upward. The median is less affected
# by extreme values, so it stays closer to the center of the smaller numbers.

# --- Hypothesis Testing Review ---

# Hypothesis Q1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat_ind, p_value_ind = stats.ttest_ind(group_a, group_b)

print("\nHypothesis Q1 - Independent samples t-test:")
print("t-statistic:", t_stat_ind)
print("p-value:", p_value_ind)


# Hypothesis Q2
print("\nHypothesis Q2 - Statistical significance at alpha = 0.05:")
if p_value_ind < 0.05:
    print("The result is statistically significant at alpha = 0.05.")
else:
    print("The result is not statistically significant at alpha = 0.05.")


# Hypothesis Q3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat_paired, p_value_paired = stats.ttest_rel(before, after)

print("\nHypothesis Q3 - Paired t-test:")
print("t-statistic:", t_stat_paired)
print("p-value:", p_value_paired)


# Hypothesis Q4
scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat_one_sample, p_value_one_sample = stats.ttest_1samp(scores, 70)

print("\nHypothesis Q4 - One-sample t-test against benchmark of 70:")
print("t-statistic:", t_stat_one_sample)
print("p-value:", p_value_one_sample)


# Hypothesis Q5
t_stat_one_tailed, p_value_one_tailed = stats.ttest_ind(
    group_a,
    group_b,
    alternative="less"
)

print("\nHypothesis Q5 - One-tailed test (group_a < group_b):")
print("p-value:", p_value_one_tailed)


# Hypothesis Q6
print("\nHypothesis Q6 - Plain-language conclusion:")
if p_value_ind < 0.05:
    print(
        "Group B scored higher on average than Group A, and this difference "
        "is unlikely to be due to chance."
    )
else:
    print(
        "Although Group B scored higher on average than Group A, the difference "
        "could reasonably be due to chance."
    )

# --- Correlation Review ---

# Correlation Q1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

correlation_matrix = np.corrcoef(x, y)

print("\nCorrelation Q1 - Full correlation matrix:")
print(correlation_matrix)

print("\nCorrelation Q1 - Correlation coefficient:")
print(correlation_matrix[0, 1])

# I expect the correlation to be 1.0 because y increases in a perfectly
# linear way as x increases, with y always equal to 2 * x.


# Correlation Q2
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]

corr_coef, p_value = pearsonr(x, y)

print("\nCorrelation Q2 - Pearson correlation coefficient:")
print(corr_coef)

print("Correlation Q2 - p-value:")
print(p_value)


# Correlation Q3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55, 60, 65, 72, 80],
    "age": [25, 30, 22, 35, 28],
}

df = pd.DataFrame(people)
correlation_df = df.corr()

print("\nCorrelation Q3 - Correlation matrix:")
print(correlation_df)


# Correlation Q4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

print("\nCorrelation Q4 - Scatter plot created.")

plt.figure()
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Correlation Q5
print("\nCorrelation Q5 - Heatmap created.")

plt.figure()
sns.heatmap(correlation_df, annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines ---

# Pipeline Q1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])


def create_series(arr):
    series = pd.Series(arr, name="values")
    return series


def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series


def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }
    return summary


def data_pipeline(arr):
    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)
    return summary


pipeline_result = data_pipeline(arr)

print("\nPipeline Q1 - Summary:")
for key, value in pipeline_result.items():
    print(f"{key}: {value}")