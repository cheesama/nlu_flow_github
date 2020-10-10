from fastapi import FastAPI

from nlu_flow.preprocessor.text_preprocessor import normalize

from utils import (
    tokenize,
    tokenize_fn,
    convert_ner_data_format,
    word2features,
    sent2features,
    sent2labels,
    sent2tokens
)

import pycrfsuite

app = FastAPI()
is_ready = False

#load domain_classifer_model
model = None
model = pycrfsuite.Tagger()
model.open('crf_entity_extractor.crfsuite')

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

@app.post("/predict")
async def predict_entities(text: str):
    feature = sent2features(convert_ner_data_format(text, entities=[]))
    entities = model.tag(feature)
    tokens = tokenize(text)

    print (tokens)
    print (entities)

    result = []
    token_value = ''
    entity_value = ''

    for i, (token, entity) in enumerate(zip(tokens, entities)):
        if entity != 'O':
            entity_value = entity.replace('B-','').replace('I-','')
            token_value += token

            if i < len(entities) - 1 and entities[i][2:] == entities[i+1][2:]:
                continue

            result.append({'entity': entity_value, 'value': token_value})
            token_value = ''
            entity_value = ''

    return {'entities': result, 'extractor': 'crf_entity_extractor.crfsuite'}


