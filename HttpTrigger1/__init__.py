import azure.functions as func
import json
from lib.model import CFRecommender

model = CFRecommender()
def main(req: func.HttpRequest) -> func.HttpResponse:
    user_id = req.get_json().get('userId')
    preds, _ = model.make_recommendation(user_id) 
    response_body = json.dumps(preds.tolist())
    return func.HttpResponse(response_body, mimetype="application/json", charset='utf-8' , status_code=200)
