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
    #name = model.predict([normalize(text, with_space=True)])[0]
    #confidence = model.predict_proba([normalize(text)])[0].max()
    probs = model.predict_proba([text])
    result = sorted( zip( model.classes_, probs[0] ), key=lambda x:x[1], reverse=True )[-top_k:]
    dict_result = []
    for each_result in result:
        each_dict = {'faq_intent': each_result[0], 'confidence': each_result[1], 'classifier': 'faq_classifier_model.rf'}
        each_dict.update(response_dict[each_result[0]])
        dict_result.append(each_dict)
        
    return dict_result

