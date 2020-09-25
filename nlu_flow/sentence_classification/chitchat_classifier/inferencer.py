from fastapi import FastAPI

from transformers import ElectraModel, ElectraTokenizer

from koelectra_fine_tuner import KoelectraQAFineTuner

import torch.nn.functional as F
import torch
import dill

app = FastAPI()
is_ready = False

#load response_dict
response_dict = {}
with open('./response_dict.dill', 'rb') as responsefile:
    response_dict = dill.load(responsefile)

#load chitchat_retrieval_model
model = None
model = KoelectraQAFineTuner(class_num=len(response_dict))
model.load_state_dict(torch.load('./koelectra_chitchat_classifier_model.modeldict', map_location=lambda storage, loc: storage))
model.eval()

#load tokenizer
MAX_LEN = 64
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

if model:
    is_ready = true

#endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {'code': 200}
    else:
        output = {'code': 500}
    return output

@app.post("/chitchat_classification/predict")
async def chitchat_response_classification(text: str):
    with torch.no_grad():
        tokens = tokenizer.encode(text, max_length=MAX_LEN, pad_to_max_length=True, truncation=True)
        feature = model(torch.tensor(tokens).unsqueeze(0))
        confidence = F.softmax(feature, dim=1).max()
        pred_class = feature.argmax(1)

    return {'confidence': confidence, 'response': response_dict[pred_class], 'model': 'koelectra_chitchat_classifier_model'}

