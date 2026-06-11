"""
Week 10 Project: LLM Transform Pipeline

Video link:
https://drive.google.com/file/d/1vgB7_4puse1Fmw9kjOatt2_MNrDzykp5/view?usp=sharing

Reflection:
Classifying weather conditions for outdoor running is a reasonable learning
example for using an LLM as a transform step, but it is probably not the best
production use case. Since the inputs are only temperature and precipitation,
deterministic rules could classify the records more cheaply, faster, and more
consistently. A rule-based approach would gain reproducibility and lower cost,
but it would lose flexibility if the classification later included more nuanced
freeform context such as wind, air quality, user preferences, or event notes.

This script reads raw weather JSON from Azure Blob Storage, reshapes hourly
weather lists into per-hour records, classifies the first 24 records with an LLM,
uploads enriched records back to Blob Storage, downloads the processed blob for
a spot-check, and saves the first 10 enriched records locally.
"""

import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from openai import OpenAI


ACCOUNT_URL = "https://evgeniiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

RAW_BLOB_PATH = f"raw/{date.today().isoformat()}/weather.json"
PROCESSED_BLOB_PATH = f"processed/{date.today().isoformat()}/weather_classified.json"

FALLBACK_PATH = Path("assignments") / "resources" / "weather_raw.json"
OUTPUTS_DIR = Path("assignments_10") / "outputs"
FIRST_10_OUTPUT_PATH = OUTPUTS_DIR / "first_10_records.json"

SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)

VALID_LABELS = {"good", "marginal", "bad"}


def get_container_client():
    """Create and return an Azure Blob Storage ContainerClient."""
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
    )
    return blob_service_client.get_container_client(CONTAINER)


def download_raw_weather(container_client) -> dict:
    """Download the Week 9 raw weather JSON, or use fallback data if needed."""
    try:
        print(f"Reading raw weather data from blob: {RAW_BLOB_PATH}")
        downloader = container_client.download_blob(RAW_BLOB_PATH)
        raw_bytes = downloader.readall()
        return json.loads(raw_bytes.decode("utf-8"))
    except ResourceNotFoundError:
        print(f"Raw blob not found at {RAW_BLOB_PATH}. Using fallback data.")
        with FALLBACK_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)


def reshape_hourly_records(weather_data: dict) -> list[dict]:
    """Convert hourly parallel lists into a list of per-hour dictionaries."""
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

    return records


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


def enrich_records(records: list[dict], limit: int = 24) -> list[dict]:
    """Classify the first limit records and add a conditions field."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Check your .env file.")

    client = OpenAI(api_key=api_key)

    enriched_records = []

    for index, record in enumerate(records[:limit], start=1):
        conditions = classify_record(client, record)

        enriched_record = record.copy()
        enriched_record["conditions"] = conditions
        enriched_records.append(enriched_record)

        if index % 6 == 0:
            print(f"Classified {index} records...")

    return enriched_records


def upload_processed_records(container_client, records: list[dict]) -> None:
    """Upload enriched records to processed/<today>/weather_classified.json."""
    data = json.dumps(records, indent=2).encode("utf-8")

    container_client.upload_blob(
        name=PROCESSED_BLOB_PATH,
        data=data,
        overwrite=True,
    )

    print(f"Uploaded processed data to {PROCESSED_BLOB_PATH} ({len(data)} bytes).")


def download_processed_records(container_client) -> list[dict]:
    """Download and parse the processed blob."""
    downloader = container_client.download_blob(PROCESSED_BLOB_PATH)
    data = downloader.readall()
    return json.loads(data.decode("utf-8"))


def save_first_10_records(records: list[dict]) -> None:
    """Save the first 10 enriched records locally for mentor inspection."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with FIRST_10_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(records[:10], file, indent=2)

    print(f"Saved first 10 records to {FIRST_10_OUTPUT_PATH}.")


def main() -> None:
    """Run the full LLM transform pipeline."""
    container_client = get_container_client()

    weather_data = download_raw_weather(container_client)
    hourly_records = reshape_hourly_records(weather_data)

    print(f"Loaded {len(hourly_records)} hourly records.")
    print("Classifying first 24 records...")

    enriched_records = enrich_records(hourly_records, limit=24)

    upload_processed_records(container_client, enriched_records)

    processed_records = download_processed_records(container_client)
    processed_df = pd.DataFrame(processed_records)

    print("\nCondition counts:")
    print(processed_df["conditions"].value_counts())

    print("\nFirst 5 rows:")
    print(processed_df.head())

    save_first_10_records(processed_records)


if __name__ == "__main__":
    main()