"""
Week 11 Warmup: Cloud ETL

This file contains conceptual answers and small code snippets for Prefect
orchestration and production pipeline patterns.
"""

from prefect import task, get_run_logger


# --- Prefect Orchestration ---


# Q1
"""
In Prefect, a @flow represents the overall pipeline or workflow. It coordinates
the order of operations, calls tasks, passes data between them, and gives the run
a visible structure in the Prefect UI.

A @task represents one observable unit of work inside the flow, such as calling
an API, transforming records, or uploading a file. Tasks can have retries,
logging, cached results, and their own success or failure states.

For a pure helper function that only converts Celsius to Fahrenheit in memory, I
would usually not decorate it with @task. It is small, deterministic, fast, and
has no I/O or external failure risk. Keeping it as a normal helper function makes
the flow simpler and avoids unnecessary Prefect overhead.
"""


# Q2
# @task(retries=3, retry_delay_seconds=30)


# Q3
"""
If the Prefect UI shows extract as Completed, transform as Failed, and load never
ran, I would open the failed flow run in the Prefect UI and then inspect the
transform task run. I would look at the task's logs, exception traceback, state,
inputs/parameters, and timing information.

I would expect to find the specific exception that caused the failure, such as an
OpenAI API error, parsing error, missing key, invalid response label, or a problem
with the input records. Because load depends on transform, I would expect load to
be skipped or not started after transform failed.
"""


# --- Production Patterns ---


# Q1
"""
response.raise_for_status() checks the HTTP response status code and raises an
exception for error responses such as 4xx or 5xx. This is better than writing
if response.status_code != 200: print("error") because the exception marks the
task as failed and lets Prefect handle retries, logs, and downstream dependency
behavior correctly.

If the API returns a 500 error and the task uses raise_for_status(), the extract
task fails, Prefect records the failure, retries can run if configured, and
downstream tasks do not run with bad or missing data.

If the task only prints "error" and continues, the pipeline may treat the task as
successful even though it did not extract valid data. Downstream tasks might then
run on None, empty data, or an invalid response, creating confusing failures
later in the pipeline.
"""


# Q2
"""
overwrite=True protects the rerun from failing because the final blob path
already exists from a previous attempt or partial run. After I fix the bug and
rerun the pipeline, the load step can safely write the corrected output to the
same final/{today}/weather_etl.json path.

Without overwrite=True, the rerun could fail during upload with a blob already
exists error, even if the new output is correct. That would make recovery less
idempotent because I would have to manually delete the old blob or generate a new
path before the pipeline could complete.
"""


# Q3
@task
def log_loaded_records(records: list, blob_path: str) -> None:
    """Log how many records were loaded to a blob path."""
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}.")