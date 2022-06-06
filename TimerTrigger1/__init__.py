import logging
import os
import pickle
import azure.functions as func
from surprise import Dataset, Reader, KNNWithMeans
from lib.utils import get_data_files
from azure.storage.blob import BlobServiceClient
from surprise.model_selection import train_test_split

blob_service_client = BlobServiceClient.from_connection_string(os.environ['AzureWebJobsStorage'])

def main(mytimer: func.TimerRequest) -> None:
    _, df_clicks = get_data_files(blob_service_client)  

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df_clicks, reader)
    # data = list(map(tuple, data.df.values.tolist()))
    train, _ = train_test_split(data, test_size=.01)

    # To use item-based cosine similarity
    sim_options = {
        "name": "cosine",
        "user_based": False,  # Compute similarities between items
    }

    model = KNNWithMeans(sim_options=sim_options)

    model.fit(train)

    model_bytes = pickle.dumps(model)
    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container='filecontainer', blob=os.environ['MODEL_PATH'])
 
    blob_client.upload_blob(model_bytes)
    logging.info('TIIIIIIIIIIIIIIIIIIIIIIIIIIIIIMER')
