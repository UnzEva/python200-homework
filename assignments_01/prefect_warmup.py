import numpy as np
import pandas as pd
from prefect import flow, task


@task
def create_series(arr):
    series = pd.Series(arr, name="values")
    return series


@task
def clean_data(series):
    cleaned_series = series.dropna()
    return cleaned_series


@task
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0],
    }
    return summary


@flow
def pipeline_flow():
    arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

    series = create_series(arr)
    cleaned_series = clean_data(series)
    summary = summarize_data(cleaned_series)

    print("Pipeline Q2 - Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    return summary


if __name__ == "__main__":
    pipeline_flow()


# Why might Prefect be more overhead than it is worth here?
# This pipeline is very small, runs quickly, and only processes a handful of values.
# Plain Python functions are easier to read and require less setup, so using Prefect
# adds extra complexity without much practical benefit for such a simple workflow.

# When could Prefect still be useful?
# Prefect becomes more useful when pipelines need scheduling, retries, logging,
# monitoring, dependency management, or orchestration across multiple steps,
# files, APIs, or databases. Even simple logic can benefit from Prefect if the
# real-world workflow runs regularly, depends on outside systems, or needs better
# visibility and reliability.