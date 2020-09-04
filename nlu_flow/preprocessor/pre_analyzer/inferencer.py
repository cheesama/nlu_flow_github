from tqdm import tqdm
from fastapi import FastAPI
from fuzzywuzzy import process, fuzz

from nlu_flow.utils import meta_db_client
from nlu_flow.preprocessor.text_preprocessor import normalize

import os, sys
import random

app = FastAPI()
is_ready = False

# load pre_analysis_dict from Meta DB
pre_analysis_dict = {}

chitchat_response_dict = {}
slang_keyword_list = []
slang_response_list = []

## storing pre-analysis data
intent_rules = meta_db_client.get("intent-rules")
for data in tqdm(intent_rules, desc="storing intent-rules"):
    pre_analysis_dict[normalize(data["utterance"])] = {
        "intent": data["intent_id"]["Intent_ID"],
    }

chitchat_rules = meta_db_client.get("nlu-chitchat-utterances")
for data in tqdm(chitchat_rules, desc="storing chitchat-rules"):
    try:
        pre_analysis_dict[normalize(data["utterance"])] = {
            "intent": "intent_chitchat",
            "chitchat_class": data["class_name"]["classes"],
        }
    except:
        print (f'check data format: {data}')

chitchat_responses = meta_db_client.get("nlu-chitchat-responses")
for data in tqdm(chitchat_responses, desc="storing chitchat-responses"):
    try:
        if data["class_name"]["classes"] not in chitchat_response_dict.keys():
            chitchat_response_dict[data["class_name"]["classes"]] = []
        chitchat_response_dict[data["class_name"]["classes"]].append(data["response"])
    except:
        print (f'check data format: {data}')

slang_rules = meta_db_client.get("nlu-slang-utterances")
for data in tqdm(slang_rules, desc="storing slang-rules"):
    pre_analysis_dict[normalize(data["utterance"])] = {
        "intent": "intent_slang",
    }
    slang_keyword_list.append(normalize(data["utterance"]))
slang_responses = meta_db_client.get("nlu-slang-responses")
for data in tqdm(slang_responses, desc="storing slang-responses"):
    slang_response_list.append(data["response"])

faq_rules = meta_db_client.get("nlu-faq-questions")
for data in tqdm(faq_rules, desc="storing faq-rules"):
    pre_analysis_dict[normalize(data["question"])] = {
        "intent": "intent_FAQ",
    }

is_ready = True

def analyze_text_with_pre_analyzer(text: str, analysis_dict: dict, lev_distance_threshold=90):
    text = normalize(text)

    ## 1. check intent_name is set directly
    if "intent_" in text:
        return {"name": text.strip(), "confidence": 1.0, "classifier": "pre_analyzer"}

    ## 2. check text exist in pre_analsis_dictionary
    if text in pre_analysis_dict:
        result = pre_analysis_dict[text]

        if "chitchat_class" in result:
            result["response"] = random.choice(
                chitchat_response_dict[result["chitchat_class"]]
            )
        elif "slang" in result["intent"]:
            result["response"] = random.choice(slang_response_list)

        result["confidence"] = 1.0
        result["classifier"] = "pre_analyzer"

        return result

    else:
        ## 2-1. check slang keyword in text
        for slang in tqdm(slang_keyword_list, desc='checking slang keywords...'):
            if slang in text:
                result = {
                    "intent": "intent_slang",
                    "confidence": 1.0,
                    "classifier": "pre_analyzer",
                    "response": random.choice(slang_response_list),
                }

                return result

        ## 3. check Levenshtein distance among given text and entire pre_analysis dictionary
        similarity_result = process.extractOne(text, pre_analysis_dict.keys())
        similarity_result_info = pre_analysis_dict[similarity_result[0]]

        ## 3-1. when chitchat is detected, choose response from same chitchat_class
        if 'chitchat_class' in similarity_result_info:
            similarity_result_info['response'] =  random.choice(chitchat_response_dict[similarity_result_info["chitchat_class"]])
        else:
            similarity_result_info['response'] = similarity_result[0]
            similarity_result_info['confidence'] = similarity_result[1] * 0.01
            similarity_result_info['lev_distance_threshold'] = lev_distance_threshold * 0.01
            similarity_result_info['classifier'] = 'pre_analyzer'

        return similarity_result_info
        
#endpoints
@app.get("/")
async def health():
    if is_ready:
        output = {'code': 200}
    else:
        output = {'code': 500}
    return output

@app.post("/pre_analyzer/predict")
async def match_pre_analyzer(text: str):
    return analyze_text_with_pre_analyzer(text, pre_analysis_dict)





