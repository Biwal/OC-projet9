import random
import os
from lib.utils import get_data_files, get_file
from surprise import Dataset, Reader
from collections import defaultdict
from azure.storage.blob import BlobServiceClient
import pickle


blob_service_client = BlobServiceClient.from_connection_string(os.environ['AzureWebJobsStorage'])

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
        # Select 5 randoms articles for each recommanded cat.
        random_articles_by_cat = [self.df_articles[self.df_articles['category_id'] == x]['article_id'].sample(5).values for x in recommanded_cat]
        #Select one of the recommanded cat and return 5 articles.
        rand_category = random.sample(random_articles_by_cat, 1)
        # rand_category =  self.df_articles['article_id'].sample(5).values
        
        return rand_category, recommanded_cat

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
        
        
        
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

class CBRecommender():
    def __init__(self) -> None:
        self.df = pd.read_csv('data/articles_by_user.csv')
        self.metadatas = pd.read_csv('data/articles_metadata.csv')
        self.embedding = pd.read_pickle('data/articles_embeddings.pickle')
        self.clicks = pd.read_csv('data/merged_clicks.csv')
        
        
        # ### Limite 16G RAM Sur poste personnel
        # self.cosine_similarities = linear_kernel(self.embedding[:50000], self.embedding[:50000])
        
        articles = self.clicks.click_article_id.value_counts().index
        self.metadatas = self.metadatas.loc[articles].reset_index()
        self.embedding = self.embedding[articles] 
        self.cosine_similarities = linear_kernel(self.embedding, self.embedding)
        
    def simScores(self, index):
        sim_scores = list(enumerate(self.cosine_similarities[index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:21]
        
        return sim_scores #warning : based on index !! not article_id
    
    def get_predictions(self, userId):
        _category_weight_matrix = pd.DataFrame(columns=['click'])
    
        #Get user favorites categories.
        _row = self.df.loc[userId]['categories']
        _row = _row.replace('[', '').replace(']', '').replace(',', '').split()
        
            #Add user favorites categories to category weight matrix.
        for index, val in pd.Series(_row).value_counts().items():
            _category_weight_matrix.loc[index] = int(val)
        
            #Normalize and format category weight matrix.
        _category_weight_matrix['click_norm'] = _category_weight_matrix.apply(lambda x : x / _category_weight_matrix['click'].max())
        _category_weight_matrix = _category_weight_matrix.reset_index()
        _category_weight_matrix = _category_weight_matrix.rename(columns={"index": "category_id"})
        _category_weight_matrix['category_id'] = _category_weight_matrix['category_id'].astype(int)

        
        #Calculate article similarities
        _article_similarity_score = []
        for index, row in _category_weight_matrix.iterrows():
            _x = self.simScores(row.category_id)
            for i in range(1, row.click + 1):
                _article_similarity_score = _article_similarity_score + _x
                
        #Building dataframe
        _recommendation_df = pd.DataFrame(columns=['index', 'article_id', 'category_id', 'sim_score', 'click_weight'])
        for row in _article_similarity_score:
            _index = row[0]
            _article_id = self.metadatas.loc[_index].article_id
            _category_id = self.metadatas.loc[_index].category_id
            if _category_id in _category_weight_matrix.category_id.values:
                _click_weight = _category_weight_matrix.loc[_category_weight_matrix.category_id == 281].click_norm.values[0] #We use normalized value of clicked, we could use non normalized too.
            else:
                _click_weight = 0
            _sim_score = row[1]
            _new_row = {'index':_index, 'article_id':_article_id, 'category_id':_category_id, 'sim_score':_sim_score, 'click_weight':_click_weight}
            _recommendation_df = _recommendation_df.append(_new_row, ignore_index=True)
            
        #Calculate final score
        _recommendation_df = _recommendation_df.assign(score = lambda x: x['sim_score'] * x['click_weight']) 
        
        _recommendation_list = np.array(_recommendation_df.sort_values(by=['score'], ascending=False).head(5).article_id.values, dtype='int')
        
        return _recommendation_list, _recommendation_df
        
        
        

            
        