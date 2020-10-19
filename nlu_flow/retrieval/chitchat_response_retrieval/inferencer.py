from fastapi import FastAPI

from transformers import ElectraModel, ElectraTokenizer

from koelectra_fine_tuner import KoelectraQAFineTuner

import torch
import faiss
import dill

app = FastAPI()
is_ready = False

#load chitchat_retrieval_model
model = None
model = KoelectraQAFineTuner()
model.load_state_dict(torch.load('./koelectra_chitchat_retrieval_model.modeldict', map_location=lambda storage, loc: storage))

#load tokenizer
MAX_LEN = 64
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

#load index
index = faiss.read_index('chitchat_retrieval_index')
top_k = 1

#load response_dict
response_dict = {}
with open('./response_dict.dill', 'rb') as responseFile:
    response_dict = dill.load(responseFile)

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

@app.post("/chitchat_retrieval/search")
async def search_chitchat_answer(text: str):
    with torch.no_grad():
        tokens = tokenizer.encode(text, max_length=MAX_LEN, pad_to_max_length=True, truncation=True)
        feature = model.get_question_feature(torch.tensor(tokens).unsqueeze(0))
        distance, neighbour = index.search(feature,k = top_k)

    return {'similarity': distance, 'response': response_dict[neighbour[0][0]], 'model': 'koelectra_chitchat_retrieval_model'}

