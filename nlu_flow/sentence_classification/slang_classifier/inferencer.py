from fastapi import FastAPI

from nlu_flow.preprocessor.text_preprocessor import normalize

import dill

app = FastAPI()
is_ready = False

#load slang_classifer_model
model = None
with open('./slang_classifier_model.rf', 'rb') as f:
    model = dill.load(f)
    print ('slang_classifier_model load success')

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

@app.post("/slang_classifier/predict")
async def predict_slang(text: str):
    name = model.predict([normalize(text)])[0]
    confidence = model.predict_proba([normalize(text)])[0].max()

    return {'label': name, 'confidence': confidence, 'Classifier': 'slang_classifier_model.rf'}

