from tqdm import tqdm

from nlu_flow.utils import meta_db_client
from nlu_flow.preprocessor.text_preprocessor import normalize

import os, sys
import random

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
    pre_analysis_dict[normalize(data["utterance"])] = {
        "intent": "intent_chitchat",
        "chitchat_class": data["class_name"]["classes"],
    }
chitchat_responses = meta_db_client.get("nlu-chitchat-responses")
for data in tqdm(chitchat_responses, desc="storing chitchat-responses"):
    if data["class_name"]["classes"] not in chitchat_response_dict.keys():
        chitchat_resposne_dict[data["class_name"]["classes"]] = []
    chitchat_resposne_dict[data["class_name"]["classes"]].append(data["response"])

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


def analyze_intent(text: str, analysis_dict: dict):
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
                    "response": random.choice(
                        chitchat_response_dict[result["chitchat_class"]]
                    ),
                }

                return result

        


