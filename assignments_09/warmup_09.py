"""
Week 9 Warmup: Data in the Cloud

This file contains conceptual answers as comments and small helper functions
for Azure authentication and Blob Storage practice.
"""

from azure.storage.blob import ContainerClient


# --- Azure Authentication ---


# Q1
"""
When I run a Python script locally that uses DefaultAzureCredential, it relies on
the credentials already available in my local development environment. In this
week's setup, that means the Azure CLI login session.

Before running the script, I must run:

    az login

DefaultAzureCredential tries several authentication methods in order. One of
those methods is AzureCliCredential, which checks whether the Azure CLI already
has an authenticated account. If I have successfully run az login, the Python
script can reuse that login without hardcoding a password or secret.
"""


# Q2
"""
A deployed pipeline running on an Azure VM, container, or other cloud service
should not use az login because there is no human sitting at the terminal to
open a browser and complete an interactive login. It would also be unsafe and
fragile to depend on a personal CLI session in production.

Instead, a deployed pipeline normally uses a managed identity or another service
identity. Azure gives that workload its own identity, and permissions can be
granted to that identity through role-based access control.

The same Python code can still work without changes because DefaultAzureCredential
checks multiple credential sources. Locally, it can use AzureCliCredential. In
Azure, it can use ManagedIdentityCredential. The script still creates
DefaultAzureCredential, but the credential source changes depending on the
environment.
"""


# Q3
"""
If a script creates DefaultAzureCredential and immediately gets an
AuthenticationError, the two most likely causes are:

1. I am not logged in locally.
   I would diagnose this by running:
       az account show
   If that fails, I would run:
       az login
   and then try the script again.

2. The script is logged in but the account or identity does not have permission
   to access the requested Azure resource.
   I would diagnose this by checking the exact error message, confirming the
   active subscription with:
       az account show
   and checking whether the signed-in user or managed identity has the needed
   role assignment, such as Reader for subscription metadata or Storage Blob Data
   Contributor for blob operations.
"""


# --- Blob Storage ---


# Q1
"""
Azure Blob Storage has a three-level hierarchy:

1. Storage account
   This is the top-level Azure resource. It is like a whole filing cabinet or
   a disk drive.

2. Container
   A container lives inside a storage account and groups related blobs. It is
   like a folder or drawer inside the filing cabinet.

3. Blob
   A blob is the actual stored object, such as a CSV file, JSON file, image, or
   model artifact. It is like a file inside the folder.

For example, a storage account might be my whole cloud drive, a container might
be a folder named raw-data, and a blob might be raw-data/weather/2026-06-09.json.
"""


# Q2
"""
Scenario 1:
I would use Blob Storage because hourly REST API JSON responses are raw files
that may need to be stored exactly as received and reprocessed later.

Scenario 2:
I would use a relational database such as Azure SQL because the analytics team
needs to query 50 million transaction rows by date range and customer ID every
day, which is a structured query workload.

Scenario 3:
I would use Blob Storage because image embeddings as NumPy arrays are file-like
artifacts that need to be saved and loaded between pipeline runs, not queried
row-by-row like relational data.
"""


# Q3
def list_container(container_client: ContainerClient) -> None:
    """Print the name and size of every blob in a container."""
    for blob in container_client.list_blobs():
        print(f"{blob.name}: {blob.size} bytes")


# Q4
def upload_text(container_client: ContainerClient, blob_name: str, text: str) -> None:
    """Upload a UTF-8 encoded string as a blob, overwriting any existing blob."""
    data = text.encode("utf-8")
    container_client.upload_blob(name=blob_name, data=data, overwrite=True)