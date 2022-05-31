import os
import pandas as pd
from typing import Tuple
from azure.storage.blob import BlobServiceClient
from io import StringIO


def get_data_files(blob_service_client: BlobServiceClient)-> Tuple[pd.DataFrame, pd.DataFrame]:
    df_articles = get_azure_csv(os.environ['ARTICLES_FILE_NAME'], blob_service_client)
    df_clicks = get_azure_csv(os.environ['CLICKS_FILE_NAME'], blob_service_client)
    df_clicks = setup_clicks_file(df_articles, df_clicks)
    
    return df_articles, df_clicks
    

def get_file(filename:str, blob_service_client:BlobServiceClient)->object:
    container_name = "filecontainer"
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=filename
    )
    return blob_client.download_blob().readall()

def get_azure_csv(
    filename: str, blob_service_client: BlobServiceClient
) -> pd.DataFrame:
    file = get_file(filename, blob_service_client)
    return pd.read_csv(StringIO(file.decode("utf-8")))
    

def setup_clicks_file(
    df_articles: pd.DataFrame, df_clicks: pd.DataFrame
) -> pd.DataFrame:
    # Create a map to convert article_id to category
    dict_article_categories = df_articles.set_index("article_id")[
        "category_id"
    ].to_dict()

    # Get Categorie associate for each article
    df_clicks["category_id"] = (
        df_clicks["click_article_id"].map(dict_article_categories).astype(int)
    )
    df_clicks["total_click"] = df_clicks.groupby(["user_id"])[
        "click_article_id"
    ].transform("count")
    df_clicks["total_click_by_category_id"] = df_clicks.groupby(
        ["user_id", "category_id"]
    )["click_article_id"].transform("count")
    df_clicks["rating"] = (
        df_clicks["total_click_by_category_id"] / df_clicks["total_click"]
    )

    return df_clicks[["user_id", "category_id", "rating"]]