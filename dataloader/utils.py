import os
from google.cloud import storage

def load_from_gcs(bucket_name, prefix=None):
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"

    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    train_from = []
    for blob in blobs:
        name = blob.name
        gsutil = 'gs://' + bucket_name + '/' + name 
        train_from.append(gsutil)
    print(train_from)