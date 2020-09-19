from fastapi import FastAPI
from kogpt2_transformers import get_kogpt2_model, get_kogpt2_tokenizer

import torch
torch.manual_seed(88)

model = None

app = FastAPI()
is_ready = False

model = get_kogpt2_model()
tokenizer = get_kogpt2_tokenizer()

if model is not None:
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
async def gererate_chitchat_repsonse(test: str):
    input_ids = tokenizer.encode(test, add_special_tokens=False, return_tensors="pt")
    generated_sequence = model.generate(input_ids=input_ids, do_sample=True, max_length=100, num_return_sequences=1)[0]
    generated_sequence = generated_sequence.tolist()
    response = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
    print("GENERATED SEQUENCE : {0}".format(response))

    return {
        "response": response,
        "generator": "KoGPT2-Transformers",
    }
