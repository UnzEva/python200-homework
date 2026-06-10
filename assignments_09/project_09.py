"""
Week 9 Project: Extract + Load Pipeline

Video link:
TODO: https://drive.google.com/file/d/1kgjPsG89HOf0mdOA28tURZT9fKR4-tCN/view?usp=sharing

This script extracts 7 days of hourly weather data from the Open-Meteo API,
serializes the response as JSON, uploads it to Azure Blob Storage, lists blobs
in the container, downloads the uploaded blob, saves a local copy, and loads the
hourly data into a pandas DataFrame.
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


# Fill in your own storage account URL before running.
ACCOUNT_URL = "https://evgeniiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

# Charlotte, NC
LATITUDE = 35.2271
LONGITUDE = -80.8431

OUTPUTS_DIR = Path("assignments_09") / "outputs"
LOCAL_OUTPUT_PATH = OUTPUTS_DIR / "weather_raw.json"


def get_container_client():
    """Create and return an Azure Blob Storage ContainerClient."""
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=ACCOUNT_URL,
        credential=credential,
    )
    return blob_service_client.get_container_client(CONTAINER)


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
    return response.json()


def serialize_weather_data(weather_data: dict) -> bytes:
    """Convert the weather API response to UTF-8 JSON bytes."""
    return json.dumps(weather_data, indent=2).encode("utf-8")


def upload_weather_json(container_client, data: bytes) -> str:
    """Upload serialized weather data to raw/<today>/weather.json."""
    today = date.today().isoformat()
    blob_path = f"raw/{today}/weather.json"

    container_client.upload_blob(
        name=blob_path,
        data=data,
        overwrite=True,
    )

    print(f"Uploaded {blob_path} ({len(data)} bytes).")
    return blob_path


def list_blobs(container_client) -> None:
    """Print all blobs in the configured container."""
    print("\nBlobs in container:")
    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")


def download_weather_json(container_client, blob_path: str) -> dict:
    """Download the uploaded JSON blob, save a local copy, and return parsed JSON."""
    downloader = container_client.download_blob(blob_path)
    downloaded_bytes = downloader.readall()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_OUTPUT_PATH.write_bytes(downloaded_bytes)

    print(f"\nDownloaded blob and saved local copy to {LOCAL_OUTPUT_PATH}.")

    return json.loads(downloaded_bytes.decode("utf-8"))


def weather_to_dataframe(weather_data: dict) -> pd.DataFrame:
    """Load the hourly field from the weather JSON into a pandas DataFrame."""
    hourly = weather_data["hourly"]
    return pd.DataFrame(hourly)


def main() -> None:
    """Run the full Extract + Load pipeline."""
    container_client = get_container_client()

    weather_data = extract_weather_data()
    serialized_data = serialize_weather_data(weather_data)

    blob_path = upload_weather_json(container_client, serialized_data)

    list_blobs(container_client)

    downloaded_data = download_weather_json(container_client, blob_path)
    weather_df = weather_to_dataframe(downloaded_data)

    print("\nFirst 5 rows of hourly weather data:")
    print(weather_df.head())


if __name__ == "__main__":
    main()