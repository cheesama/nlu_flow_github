from fastapi import FastAPI

from nlu_flow.preprocessor.text_preprocessor import normalize

import dill
import random

app = FastAPI()
is_ready = False

# load chitchat_classifer_model
model = None
with open("./chitchat_classifier_model.svc", "rb") as f:
    model = dill.load(f)
    print("chitchat_classifier_model load success")

response_dict = {}
with open("response_dict.dill", "rb") as responseFile:
    response_dict = dill.load(responseFile)

if model:
    is_ready = True

# endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {"code": 200}
    else:
        output = {"code": 500}
    return output


@app.post("/chitchat_classifier/predict")
async def chitchat_response(text: str):
    name = model.predict([normalize(text)])[0]
    confidence = model.predict_proba([normalize(text)])[0].max()

    return {
        "class": name,
        "confidence": confidence,
        "response": random.choice(response_dict[name]),
        "Classifier": "chitchat_classifier_model.svc",
    }
