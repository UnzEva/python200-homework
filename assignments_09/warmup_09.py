"""
Week 9 Warmup: Data in the Cloud

This file contains conceptual answers as comments and small helper functions
for Azure authentication and Blob Storage practice.
"""

from azure.storage.blob import ContainerClient


# --- Azure Authentication ---


# Q1
"""
DefaultAzureCredential is a chained credential. It tries several credential
providers in order until one works. In local development, the most important one
for this assignment is AzureCliCredential.

AzureCliCredential allows the Python Azure SDK to reuse the session from
`az login`, so the script can authenticate without hard-coding secrets in the code.

This is why running `az login` first matters. After I log in through the Azure
CLI, DefaultAzureCredential can find that Azure CLI session through
AzureCliCredential and use it to access Azure resources.
"""


# Q2
"""
In production, I would not rely on my personal Azure CLI login. 
A deployed pipeline should use a managed identity or another service identity. 
Permissions can then be granted to that identity through Azure role-based access control.

For example, I could assign the Storage Blob Data Contributor role to the managed identity on the storage account 
so the deployed pipeline can read and write blobs. This is safer than storing credentials in code 
because access is managed by Azure and can be changed or revoked centrally.
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

2. A second true authentication cause could be that the managed identity is not enabled on the compute resource, 
or environment variables such as AZURE_CLIENT_ID, AZURE_TENANT_ID, or AZURE_CLIENT_SECRET are set incorrectly. 
In that case, DefaultAzureCredential may try a specific credential provider but fail before it ever receives a valid token.

If authentication succeeds but the account does not have permission to the
storage account, that is an authorization problem instead. It would usually
appear as a 403 Forbidden or a permissions error after the credential is created.
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