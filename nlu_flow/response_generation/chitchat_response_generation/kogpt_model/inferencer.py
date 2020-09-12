from fastapi import FastAPI
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer

import torch

app = FastAPI()
is_ready = False

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path, num_best=0, alpha=0)

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


@app.post("/response_generator/generate")
async def gererate_chitchat_repsonse(sent: str):
    question = sent
    max_len = 128

    toked = tok(sent)

    while True:
        input_ids = torch.tensor([vocab[vocab.bos_token],] + vocab[toked]).unsqueeze(0)
        pred = model(input_ids)[0]
        gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]

        if gen == "</s>":
            break
        sent += gen.replace("▁", " ")
        toked = tok(sent)

        print (sent)

        if len(sent) > max_len:
            break

    return {
        "response": sent.replace(question, ''),
        "generator": "KoGPT2",
    }
