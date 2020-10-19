from fastapi import FastAPI

from nlu_flow.preprocessor.text_preprocessor import normalize

import dill

app = FastAPI()
is_ready = False

#load faq_classifer_model
model = None
response_dict = None
with open('./faq_classifier_model.svc', 'rb') as f:
    model = dill.load(f)
    print ('faq_classifier_model load success')

with open('./faq_response_dict.dill', 'rb') as f:
    response_dict = dill.load(f)
    print ('faq response data load success')

if model and response_dict:
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
    probs = model.predict_proba([normalize(text)])
    result = sorted( zip( model.classes_, probs[0] ), key=lambda x:x[1] )[-top_k:]
    dict_result = []
    for each_result in result:
        each_dict = {'faq_intent': each_result[0], 'confidence': each_result[1], 'classifier': 'faq_classifier_model.svc'}
        each_dict.update(response_dict[each_result[0]])
        dict_result.append(each_dict)
        
    return dict_result

