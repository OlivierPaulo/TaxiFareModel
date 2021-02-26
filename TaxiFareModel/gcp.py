import os

from google.cloud import storage
from termcolor import colored

### GCP Parameters ### 
BUCKET_NAME = "wagon-ta-ml-paulo-01"  # ⚠️ replace with your BUCKET NAME
MODEL_NAME = 'taxi_fare_model'
MODEL_VERSION = 'v1'
STORAGE_lOCATION = 'models'
FILENAME = "model.joblib"

def storage_upload(model_directory=MODEL_VERSION, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        MODEL_NAME,
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')
