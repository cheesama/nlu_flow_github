from fastapi import FastAPI
from pytorch_lightning import Trainer

from train_torch import KoGPT2Chat

import argparse
import uvicorn

import gluonnlp as nlp
import numpy as np
import pandas as pd
import torch
torch.manual_seed(88)

model = None

app = FastAPI()
is_ready = False

# load model
parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')
parser.add_argument('--model_params', type=str, default='model_chp/model_last.ckpt', help='model binary for starting chat')

parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args, unknown = parser.parse_known_args()

model = KoGPT2Chat.load_from_checkpoint(args.model_params)

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


@app.post("/predict")
async def gererate_chitchat_repsonse(text: str):
    return {
        "response": model.inference(text),
        "generator": "KoGPT2",
    }

uvicorn.run(app, host='0.0.0.0', port=8000) 
