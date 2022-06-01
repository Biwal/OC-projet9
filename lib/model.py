import random
import os
from lib.utils import get_data_files, get_file
from surprise import Dataset, Reader
from collections import defaultdict
from azure.storage.blob import BlobServiceClient
import pickle


blob_service_client = BlobServiceClient.from_connection_string(os.environ['STORAGE'])

class CFRecommender:
    def __init__(self):
        self.df_articles, self.df_clicks = get_data_files(blob_service_client)  
              
        model_bytes = get_file(os.environ['MODEL_PATH'], blob_service_client)
        self.model = pickle.loads(model_bytes)
        
        self.preds = self._get_preds()        
        
    def _get_preds(self):
        reader = Reader(rating_scale=(0, 1))
        X = Dataset.load_from_df(self.df_clicks, reader)
        X = list(map(tuple, X.df.values.tolist()))
        return self.model.test(X)
                
    #Make simple recommendation for user.
    def make_recommendation(self,user_ID:int):
        top_n = self.get_top_n()
        
        #Get top 5 cat and adding it to our list
        recommanded_cat = [iid for iid, _ in top_n[user_ID]]
        #If we don't have any recommandation, use our data.
        if not recommanded_cat:
            recommanded_cat = self.df_clicks[self.df_clicks['user_id'] == user_ID].nlargest(1, ['rating'])['category_id'].values
        #Select 5 randoms articles for each recommanded cat.
        random_articles_by_cat = [self.df_articles[self.df_articles['category_id'] == x]['article_id'].sample(5).values for x in recommanded_cat]
        #Select one of the recommanded cat and return 5 articles.
        rand_category = random.sample(random_articles_by_cat, 1)
        return rand_category[0], recommanded_cat

    #Function from https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
    def get_top_n(self,  n=5):
        # First map the predictions to each user.
        top_n = defaultdict(list)
        for uid, iid, true_r, est, _ in self.preds:
            top_n[uid].append((iid, est))
        # Then sort the predictions for each user and retrieve the k highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n    
        

            
        