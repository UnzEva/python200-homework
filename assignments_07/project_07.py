import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from dotenv import load_dotenv
from scipy.stats import pearsonr
from smolagents import CodeAgent, OpenAIServerModel, tool


if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

api_key = os.getenv("OPENAI_API_KEY")

DATA_PATH = Path("assignments_01/outputs/merged_happiness.csv")
FALLBACK_DATA_DIR = Path("assignments/resources/happiness_project")
OUTPUTS_DIR = Path("assignments_07/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

df = None


# --- Task 1: Define Your Tools ---

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged World Happiness CSV from DATA_PATH. If that file does not
    exist, falls back to loading and merging yearly CSV files from
    assignments/resources/happiness_project/.

    Returns:
        A dictionary containing the loaded DataFrame shape and column names, or
        an error dictionary if no data file can be found.
    """
    global df

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        return {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "source": str(DATA_PATH),
        }

    if not FALLBACK_DATA_DIR.exists():
        return {
            "error": (
                f"Could not find {DATA_PATH} or fallback directory "
                f"{FALLBACK_DATA_DIR}."
            )
        }

    yearly_frames = []

    for csv_file in sorted(FALLBACK_DATA_DIR.glob("*.csv")):
        year_text = csv_file.stem
        year_digits = "".join(character for character in year_text if character.isdigit())

        if not year_digits:
            continue

        year = int(year_digits)
        yearly_df = pd.read_csv(csv_file)
        yearly_df["year"] = year
        yearly_frames.append(yearly_df)

    if not yearly_frames:
        return {"error": f"No yearly CSV files found in {FALLBACK_DATA_DIR}."}

    df = pd.concat(yearly_frames, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "source": str(FALLBACK_DATA_DIR),
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    Args:
        column: The name of the column to summarize.

    Returns:
        A dictionary of descriptive statistics from pandas describe(), or an
        error dictionary if the data is not loaded or the column does not exist.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found.",
            "available_columns": df.columns.tolist(),
        }

    summary = df[column].describe().to_dict()

    cleaned_summary = {}
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            cleaned_summary[key] = round(float(value), 4)
        else:
            cleaned_summary[key] = value

    return cleaned_summary


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    Args:
        col1: The first numeric column name.
        col2: The second numeric column name.

    Returns:
        A dictionary with the column names, Pearson correlation coefficient, and
        p-value rounded to four decimal places, or an error dictionary on bad
        input.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    missing_columns = [column for column in [col1, col2] if column not in df.columns]

    if missing_columns:
        return {
            "error": f"Columns not found: {missing_columns}",
            "available_columns": df.columns.tolist(),
        }

    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return {"error": f"Both columns must be numeric. Received '{col1}' and '{col2}'."}

    clean_data = df[[col1, col2]].dropna()

    if clean_data.empty:
        return {"error": f"No valid rows available for '{col1}' and '{col2}'."}

    pearson_r, p_value = pearsonr(clean_data[col1], clean_data[col2])

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(pearson_r), 4),
        "p_value": round(float(p_value), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column for a specific year.

    Args:
        column: The column to rank countries by.
        year: The year to filter the dataset by.
        n: The number of top countries to return.

    Returns:
        A dictionary with the year, ranking column, and top countries as a list
        of dictionaries. Each country dictionary includes "country" and the
        requested column value. Returns an error dictionary on bad input.
    """
    if df is None:
        return {"error": "No data is loaded. Call load_happiness_data first."}

    if column not in df.columns:
        return {
            "error": f"Column '{column}' not found.",
            "available_columns": df.columns.tolist(),
        }

    if "year" not in df.columns:
        return {"error": "The dataset does not contain a 'year' column."}

    if "Country" not in df.columns:
        return {"error": "The dataset does not contain a 'Country' column."}

    year_data = df[df["year"] == year]

    if year_data.empty:
        return {
            "error": f"No data found for year {year}.",
            "available_years": sorted(df["year"].dropna().unique().tolist()),
        }

    top_rows = year_data.sort_values(column, ascending=False).head(n)

    countries = []

    for _, row in top_rows.iterrows():
        countries.append(
            {
                "country": row["Country"],
                column: round(float(row[column]), 4)
                if isinstance(row[column], (int, float))
                else row[column],
            }
        )

    return {
        "year": year,
        "column": column,
        "top_countries": countries,
    }

# --- Task 2: Build the Agent ---

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).

The dataset is available at assignments_01/outputs/merged_happiness.csv.
Save generated project plots to assignments_07/outputs/.
If a user says outputs/, treat that as assignments_07/outputs/.
When writing matplotlib code, use a non-GUI backend and save figures instead of displaying them.
For custom plots, you may read the CSV directly from assignments_01/outputs/merged_happiness.csv.
Do not treat the return value of load_happiness_data() as a pandas DataFrame; that tool returns metadata.

The actual dataset columns use title-style names such as:
- Happiness score
- GDP per capita
- Regional indicator
- Country
- year

If the user writes snake_case names such as happiness_score or gdp_per_capita,
map them to the actual column names before calling tools.

Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries,
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
    verbosity_level=0,
)

def run_tool_tests() -> None:
    """Run quick checks for the four project tools."""
    print("--- Mini-Project: World Happiness Agent ---")
    print("Task 1: Testing tools")

    load_result = load_happiness_data()
    print("Load result:")
    print(load_result)

    summary_result = summarize_column("Happiness score")
    print("\nSummary result:")
    print(summary_result)

    correlation_result = compute_correlation("Happiness score", "GDP per capita")
    print("\nCorrelation result:")
    print(correlation_result)

    top_countries_result = get_top_n_countries("Happiness score", 2019, 5)
    print("\nTop countries result:")
    print(top_countries_result)


def run_guided_queries() -> None:
    """Run the five required guided queries."""
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    print("\n--- Task 3: Run Guided Queries ---")

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    expected_plot_path = OUTPUTS_DIR / "happiness_by_region.png"
    print(f"\nPlot saved check: {expected_plot_path.exists()} at {expected_plot_path}")


def run_custom_queries() -> None:
    """Run two additional project queries."""
    print("\n--- Task 4: Your Own Questions ---")

    my_query_1 = "Show me the top 10 countries by GDP per capita in 2020."
    print(f"\n--- My Query 1: {my_query_1} ---")
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment:
    # I expected this query to trigger tool use because get_top_n_countries can
    # rank countries by a chosen column for a chosen year. The agent should map
    # "GDP per capita" to the correct dataset column and call the ranking tool.

    my_query_2 = (
        "Create a bar chart of the average happiness_score by region for 2020. "
        "Save it to outputs/happiness_2020_by_region_bar.png."
    )
    print(f"\n--- My Query 2: {my_query_2} ---")
    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment:
    # I expected this query to trigger code generation because none of the tools
    # can create a grouped regional bar chart. The agent needs to write pandas
    # and matplotlib code to group by region, compute average happiness score,
    # create the plot, and save it to the outputs directory.


if __name__ == "__main__":
    run_tool_tests()
    print("\nTask 2: Agent built successfully.")
    run_guided_queries()
    run_custom_queries()

# --- Reflection ---
#
# 1. In Query 3, the agent reported the Pearson correlation and p-value, then
#    said whether the result was statistically significant. It used the p-value
#    correctly: the p-value was 0.0, which is below the common threshold of 0.05,
#    so the agent treated the relationship between GDP per capita and happiness
#    score as statistically significant.
#
# 2. One response that surprised me was the plot query. The agent was able to
#    move beyond the predefined tools and write custom pandas/matplotlib code to
#    create a multi-line regional plot. Earlier, it made mistakes when treating
#    tool output like a DataFrame, so it was interesting to see that after the
#    prompt was improved, it handled the custom plotting task much better.
#
# 3. One additional useful tool would be compare_countries(country_names,
#    columns, year). It would filter the dataset to a list of countries for a
#    specific year and return selected metrics side by side. This would help
#    answer questions like "How did Finland, Denmark, and the United States
#    compare in happiness, GDP, social support, and life expectancy in 2020?"
