from azure.storage.blob import ContainerClient, BlobClient
import os

# Your SAS container URL
SAS_URL = "https://azwtewebsitecache.blob.core.windows.net/webvia?sp=rcwl&st=2025-05-05T17:40:16Z&se=2025-11-05T18:40:16Z&spr=https&sv=2024-11-04&sr=c&sig=6eTcYmq%2BeauVioFmi1bxh%2Bd4gDjvNdq54EufmpPSKYY%3D"

# Initialize container client
container_client = ContainerClient.from_container_url(SAS_URL)

# Download a single blob
def download_file(blob_name, download_dir="."):
    blob_client = container_client.get_blob_client(blob_name)
    download_path = os.path.join(download_dir, blob_name)

    # Create subdirectories if needed
    os.makedirs(os.path.dirname(download_path), exist_ok=True)

    with open(download_path, "wb") as f:
        data = blob_client.download_blob()
        f.write(data.readall())
    print(f"Downloaded {blob_name} to {download_path}")

# Download all blobs in the container
def download_all_blobs(download_dir="."):
    blobs = container_client.list_blobs()
    for blob in blobs:
        download_file(blob.name, download_dir)

# Example usage
download_all_blobs(download_dir="downloaded_blobs")
