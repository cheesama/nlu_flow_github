from fastapi import FastAPI

import dill

app = FastAPI()
is_ready = False

#load domain_classifer_model
model = None
with open('./domain_classifier_model.svc', 'rb') as f:
    model = dill.load(f)
    print ('domain_classifier_model load success')

if model:
    is_ready = True

#endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {'code': 200}
    else:
        output = {'code': 500}
    return output

@app.post("/domain_classifier/predict")
async def predict_domain(text: str):
    name = model.predict([text])[0]
    confidence = model.predict_proba([text])[0].max()

    return {'name': name, 'confidence': confidence, 'Classifier': 'domain_classifier_model.svc'}

