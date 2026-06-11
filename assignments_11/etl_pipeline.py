"""
Week 11 Capstone: Cloud ETL Pipeline

Video link:
TODO: paste video link here after recording.

This script runs a complete Prefect-orchestrated ETL pipeline:
1. Extract weather data from the Open-Meteo API.
2. Transform the first 24 hourly records with an OpenAI classification step.
3. Load the enriched records to Azure Blob Storage.
"""

import json
import os
from datetime import date

import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI
from prefect import flow, task


ACCOUNT_URL = "https://evgeniiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# Charlotte, NC
LATITUDE = 35.2271
LONGITUDE = -80.8431

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


@task(retries=2, retry_delay_seconds=10)
def extract_weather_data(latitude: float = LATITUDE, longitude: float = LONGITUDE) -> dict:
    """Extract 7 days of hourly weather data from the Open-Meteo API."""
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}"
        f"&longitude={longitude}"
        "&hourly=temperature_2m,precipitation"
        "&forecast_days=7"
    )

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    weather_data = response.json()
    print("Extracted weather data from Open-Meteo API.")

    return weather_data


def classify_record(client: OpenAI, record: dict) -> str:
    """Classify one hourly weather record for outdoor running."""
    user_message = (
        f"Temperature: {record['temperature_2m']}C, "
        f"Precipitation: {record['precipitation']}mm"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0,
    )

    label = response.choices[0].message.content.strip().lower()

    if label not in VALID_LABELS:
        return "unknown"

    return label


@task
def transform_weather_data(weather_data: dict) -> list[dict]:
    """Reshape hourly weather data and classify the first 24 records."""
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

    client = OpenAI(api_key=api_key)

    hourly = weather_data["hourly"]

    records = []
    for time_value, temperature, precipitation in zip(
        hourly["time"],
        hourly["temperature_2m"],
        hourly["precipitation"],
    ):
        records.append(
            {
                "time": time_value,
                "temperature_2m": temperature,
                "precipitation": precipitation,
            }
        )

    enriched_records = []

    for index, record in enumerate(records[:24], start=1):
        conditions = classify_record(client, record)

        enriched_record = record.copy()
        enriched_record["conditions"] = conditions
        enriched_records.append(enriched_record)

        if index % 6 == 0:
            print(f"Classified {index} records...")

    print(f"Transformed {len(enriched_records)} records.")

    return enriched_records


@task
def load_weather_data(enriched_records: list[dict]) -> str:
    """Load enriched weather records to Azure Blob Storage."""
    today = date.today().isoformat()
    blob_path = f"final/{today}/weather_etl.json"

    data = json.dumps(enriched_records, indent=2).encode("utf-8")

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
    )
    container_client = blob_service_client.get_container_client(CONTAINER)

    container_client.upload_blob(
        name=blob_path,
        data=data,
        overwrite=True,
    )

    print(f"Uploaded {blob_path} ({len(data)} bytes).")

    return blob_path


@flow(log_prints=True)
def weather_etl_flow() -> str:
    """Run the full Extract, Transform, Load cloud pipeline."""
    weather_data = extract_weather_data()
    enriched_records = transform_weather_data(weather_data)
    final_blob_path = load_weather_data(enriched_records)

    print(f"Pipeline completed successfully. Final blob path: {final_blob_path}")

    return final_blob_path


if __name__ == "__main__":
    weather_etl_flow()