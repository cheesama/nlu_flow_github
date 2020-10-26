from fastapi import FastAPI

from model import KoelectraFAQClassifier

import dill

app = FastAPI()
is_ready = False

#load faq_classifer_model
model = None
model = KoelectraFAQClassifier.load_from_checkpoint('koelectra_faq_classifier.ckpt')

if model is not None:
    is_ready = True

#endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {'code': 200}
    else:
        output = {'code': 500}
    return output

@app.post("/predict")
async def predict_faq(text: str, top_k=3):
    return model.inference(text, top_k)

