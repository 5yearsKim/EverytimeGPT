import os
from google.cloud import storage

def load_from_gcs():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/onion/private/languagemodel-tpu-key.json"

    storage_client = storage.Client()
    blobs = storage_client.list_blobs('nlp-pololo')

    train_from = []
    bucket_name = 'nlp-pololo'
    for blob in blobs:
        name = blob.name
        gsutil = 'gs://' + bucket_name + '/' + name 
        train_from.append(gsutil)
    print(train_from)