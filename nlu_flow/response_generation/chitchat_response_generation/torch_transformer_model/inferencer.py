from fastapi import FastAPI

from embedding_transformer import EmbeddingTransformer

from nlu_flow.preprocessor.text_preprocessor import normalize
from nlu_flow.utils.kor_char_tokenizer import KorCharTokenizer

import torch

app = FastAPI()
is_ready = False

# load chitchat_response_model
model = None

tokenizer = KorCharTokenizer()
model = EmbeddingTransformer(
    vocab_size=tokenizer.get_vocab_size(),
    max_seq_len=tokenizer.get_seq_len(),
    pad_token_id=tokenizer.get_pad_token_id(),
)
model.load_state_dict(torch.load('transformer_chitchat_response_model.modeldict',  map_location=lambda storage, loc: storage))

if model is not None:
    is_ready = True

print ('model load success')

# endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {"code": 200}
    else:
        output = {"code": 500}
    return output


@app.post("/chitchat_response_generator/generate")
async def generate_response(text: str):
    max_len = model.max_seq_len
    tokens = tokenizer.tokenize(text)

    while True:
        pred = model(torch.LongTensor(tokens).unsqueeze(0))
        pred = pred.argmax(2)[0].numpy()

        if len(tokens) >= tokenizer.max_len:
            break

        if pred[len(tokens) + 1] == 1: #1 means EOS token
            break

        tokens += pred[len(tokens)]

    response = tokenizer.decode(pred)

    return {
        "response": response,
        "Generator": "transformer_chitchat_response_generator.modeldict",
    }
