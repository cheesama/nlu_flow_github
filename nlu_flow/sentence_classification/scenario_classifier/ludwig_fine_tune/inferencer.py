from fastapi import FastAPI
from ludwig.api import LudwigModel

import uvicorn
import dill

app = FastAPI()
is_ready = False

#load scenario_classifer_model
model = None
model = LudwigModel.load("results/experiment_run/model")

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
async def predict_scenario(text: str, top_k=3):
    result = model.predict({'text': [text]})[0]
    result_dict = {}

    for column in result.columns:
        if 'class_probabilities_' not in column or 'UNK' in column:
            continue

        result_dict[column[column.rfind('_') + 1:]] = result[column][0]

    result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)}

    result = []
    for  k, v in result_dict.items():
        result.append({'domain': k, 'confidence': v})
        if len(result) >= int(top_k):
            break

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
