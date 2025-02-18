import os
import time
from dotenv import load_dotenv
from groundx import Document, GroundX
from natsort import natsorted

load_dotenv()

groundx_api_key = os.getenv("GROUNDX_API_KEY")
bucket_id_spanish = os.getenv("GROUNDX_BUCKET_ID_SPANISH")


def ingestDocument(file_name, file_type, upload_path, search_data):
    """
    Function to ingest a document using GroundX API.
    """
    client = GroundX(api_key=groundx_api_key)

    ingest_response = client.ingest(
        documents=[
            Document(
                bucket_id=bucket_id_spanish,
                file_name=file_name,
                file_path=upload_path,
                file_type=file_type,
                search_data=search_data
            )
        ]
    )

    # Wait for the ingestion to complete
    while True:
        ingest_response = client.documents.get_processing_status_by_id(
            process_id=ingest_response.ingest.process_id,
        )
        if ingest_response.ingest.status in ["complete", "cancelled"]:
            break
        if ingest_response.ingest.status == "error":
            raise ValueError(f"Error Ingesting Document: {file_name}")
        print(f"{file_name}: Status = {ingest_response.ingest.status}")
        time.sleep(3)

    return ingest_response


if __name__ == "__main__":
    directory = "static/docs/pdf"
    file_type = "pdf"
    search_data = None

    # Get all files in the directory and sort them naturally
    file_list = natsorted(os.listdir(directory))

    # Iterate through sorted PDF files
    for file_name in file_list:
        if file_name.endswith(".pdf"):
            upload_path = os.path.join(directory, file_name)
            try:
                print(f"Starting ingestion for: {file_name}")
                response = ingestDocument(file_name, file_type, upload_path, search_data)
                print(f"Completed ingestion for: {file_name}")
                print(f"Server response: {response}")
            except Exception as e:
                print(f"Failed to process {file_name}: {e}")
