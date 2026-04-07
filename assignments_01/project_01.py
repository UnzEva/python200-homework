from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from prefect import flow, task, get_run_logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "resources" / "happiness_project"
OUTPUT_DIR = PROJECT_ROOT / "assignments_01" / "outputs"

# --- Task 1: Load Multiple Years of Data ---
@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()

    dataframes = []

    for year in range(2015, 2025):
        file_path = DATA_DIR / f"world_happiness_{year}.csv"
        logger.info(f"Loading file: {file_path.name}")

        df = pd.read_csv(file_path, sep=";", decimal=",")

        if "Ladder score" in df.columns:
            df = df.rename(columns={"Ladder score": "Happiness score"})

        #logger.info(f"{year} columns: {list(df.columns)}")
        df["year"] = year
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)

    output_path = OUTPUT_DIR / "merged_happiness.csv"
    merged_df.to_csv(output_path, index=False)

    logger.info(f"Merged dataset saved to: {output_path}")

    return merged_df

# --- Task 2: Descriptive Statistics ---
@task
def descriptive_statistics(merged_df):
    logger = get_run_logger()

    mean_score = merged_df["Happiness score"].mean()
    median_score = merged_df["Happiness score"].median()
    std_score = merged_df["Happiness score"].std()

    logger.info(f"Overall mean happiness score: {mean_score:.3f}")
    logger.info(f"Overall median happiness score: {median_score:.3f}")
    logger.info(f"Overall standard deviation of happiness score: {std_score:.3f}")

    mean_by_year = merged_df.groupby("year")["Happiness score"].mean()
    logger.info("Mean happiness score by year:")
    for year, value in mean_by_year.items():
        logger.info(f"{year}: {value:.3f}")

    mean_by_region = merged_df.groupby("Regional indicator")["Happiness score"].mean().sort_values(ascending=False)
    logger.info("Mean happiness score by region:")
    for region, value in mean_by_region.items():
        logger.info(f"{region}: {value:.3f}")

    return {
        "overall_mean": mean_score,
        "overall_median": median_score,
        "overall_std": std_score,
        "mean_by_year": mean_by_year,
        "mean_by_region": mean_by_region,
    }

# --- Task 3: Visual Exploration ---
@task
def create_visualizations(merged_df):
    logger = get_run_logger()

    plt.figure()
    merged_df["Happiness score"].hist(bins=20)
    plt.title("Happiness Score Distribution")
    plt.xlabel("Happiness score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "happiness_histogram.png")
    plt.close()
    logger.info("Saved plot: happiness_histogram.png")

    plt.figure(figsize=(10, 6))
    merged_df.boxplot(column="Happiness score", by="year")
    plt.title("Happiness Score by Year")
    plt.suptitle("")
    plt.xlabel("Year")
    plt.ylabel("Happiness score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "happiness_by_year.png")
    plt.close()
    logger.info("Saved plot: happiness_by_year.png")

    plt.figure()
    plt.scatter(merged_df["GDP per capita"], merged_df["Happiness score"])
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per capita")
    plt.ylabel("Happiness score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gdp_vs_happiness.png")
    plt.close()
    logger.info("Saved plot: gdp_vs_happiness.png")

    numeric_df = merged_df.select_dtypes(include="number")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    plt.close()
    logger.info("Saved plot: correlation_heatmap.png")

# --- Task 4: Hypothesis Testing ---
@task
def hypothesis_testing(merged_df):
    logger = get_run_logger()

    scores_2019 = merged_df[merged_df["year"] == 2019]["Happiness score"].dropna()
    scores_2020 = merged_df[merged_df["year"] == 2020]["Happiness score"].dropna()

    t_stat_2019_2020, p_value_2019_2020 = stats.ttest_ind(scores_2019, scores_2020)

    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()

    logger.info("2019 vs 2020 independent samples t-test:")
    logger.info(f"t-statistic: {t_stat_2019_2020:.4f}")
    logger.info(f"p-value: {p_value_2019_2020:.4f}")
    logger.info(f"Mean happiness score for 2019: {mean_2019:.4f}")
    logger.info(f"Mean happiness score for 2020: {mean_2020:.4f}")

    if p_value_2019_2020 < 0.05:
        logger.info(
            "The difference in happiness scores between 2019 and 2020 is "
            "statistically significant and is unlikely to be due to chance."
        )
    else:
        logger.info(
            "The difference in happiness scores between 2019 and 2020 is "
            "not statistically significant and could reasonably be due to chance."
        )

    region_a = merged_df[merged_df["Regional indicator"] == "North America and ANZ"]["Happiness score"].dropna()
    region_b = merged_df[merged_df["Regional indicator"] == "Sub-Saharan Africa"]["Happiness score"].dropna()

    t_stat_regions, p_value_regions = stats.ttest_ind(region_a, region_b)

    logger.info("North America and ANZ vs Sub-Saharan Africa independent samples t-test:")
    logger.info(f"t-statistic: {t_stat_regions:.4f}")
    logger.info(f"p-value: {p_value_regions:.4f}")

    return {
        "2019_2020_t_stat": t_stat_2019_2020,
        "2019_2020_p_value": p_value_2019_2020,
        "2019_mean": mean_2019,
        "2020_mean": mean_2020,
        "region_t_stat": t_stat_regions,
        "region_p_value": p_value_regions,
    }

# --- Task 5: Correlation and Multiple Comparisons ---
@task
def correlation_analysis(merged_df):
    logger = get_run_logger()

    numeric_columns = merged_df.select_dtypes(include="number").columns.tolist()
    explanatory_columns = [
        column for column in numeric_columns
        if column not in ["Happiness score", "year", "Ranking"]
    ]

    correlation_results = []

    for column in explanatory_columns:
        clean_df = merged_df[["Happiness score", column]].dropna()
        corr_coef, p_value = stats.pearsonr(clean_df["Happiness score"], clean_df[column])

        logger.info(
            f"{column}: correlation = {corr_coef:.4f}, p-value = {p_value:.6f}"
        )

        correlation_results.append({
            "variable": column,
            "correlation": corr_coef,
            "p_value": p_value,
        })

    number_of_tests = len(correlation_results)
    adjusted_alpha = 0.05 / number_of_tests

    logger.info(f"Number of correlation tests: {number_of_tests}")
    logger.info(f"Bonferroni-adjusted alpha: {adjusted_alpha:.6f}")

    for result in correlation_results:
        if result["p_value"] < 0.05:
            logger.info(
                f"{result['variable']} is significant at alpha = 0.05"
            )

    for result in correlation_results:
        if result["p_value"] < adjusted_alpha:
            logger.info(
                f"{result['variable']} remains significant after Bonferroni correction"
            )

    return {
        "correlation_results": correlation_results,
        "number_of_tests": number_of_tests,
        "adjusted_alpha": adjusted_alpha,
    }

# --- Task 6: Summary Report ---

@task
def summary_report(merged_df, hypothesis_results, correlation_results):
    logger = get_run_logger()

    total_countries = merged_df["Country"].nunique()
    total_years = merged_df["year"].nunique()

    region_means = (
        merged_df.groupby("Regional indicator")["Happiness score"]
        .mean()
        .sort_values(ascending=False)
    )

    top_3_regions = region_means.head(3)
    bottom_3_regions = region_means.tail(3)

    logger.info(f"Total number of countries: {total_countries}")
    logger.info(f"Total number of years: {total_years}")
    logger.info(f"Top 3 regions by mean happiness score: {top_3_regions.to_dict()}")
    logger.info(f"Bottom 3 regions by mean happiness score: {bottom_3_regions.to_dict()}")

    if hypothesis_results["2019_2020_p_value"] < 0.05:
        logger.info(
            "The 2019 vs 2020 t-test suggests a statistically significant change "
            "in happiness scores between the pre-pandemic and pandemic period."
        )
    else:
        logger.info(
            "The 2019 vs 2020 t-test does not show a statistically significant "
            "change in happiness scores between the pre-pandemic and pandemic period."
        )

    adjusted_alpha = correlation_results["adjusted_alpha"]
    significant_results = [
        result
        for result in correlation_results["correlation_results"]
        if result["p_value"] < adjusted_alpha
    ]

    if significant_results:
        strongest_result = max(
            significant_results,
            key=lambda result: abs(result["correlation"])
        )
        logger.info(
            "Most strongly correlated variable with happiness score after "
            f"Bonferroni correction: {strongest_result['variable']} "
            f"(correlation = {strongest_result['correlation']:.4f})"
        )
    else:
        logger.info(
            "No variables remained significant after Bonferroni correction."
        )

@flow
def happiness_pipeline():
    merged_df = load_and_merge_data()
    descriptive_statistics(merged_df)
    create_visualizations(merged_df)
    hypothesis_results = hypothesis_testing(merged_df)
    correlation_results = correlation_analysis(merged_df)
    summary_report(merged_df, hypothesis_results, correlation_results)

if __name__ == "__main__":
    happiness_pipeline()