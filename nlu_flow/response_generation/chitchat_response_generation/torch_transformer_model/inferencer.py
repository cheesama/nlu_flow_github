from fastapi import FastAPI

from nlu_flow.preprocessor.text_preprocessor import normalize

import torch

app = FastAPI()
is_ready = False

#load domain_classifer_model
model = None

#endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {'code': 200}
    else:
        output = {'code': 500}
    return output

@app.post("/chitchat_response_generator/generate")
async def predict_domain(text: str):
    name = model.predict([normalize(text)])[0]
    confidence = model.predict_proba([normalize(text)])[0].max()

    return {'name': name, 'confidence': confidence, 'Classifier': 'domain_classifier_model.svc'}

