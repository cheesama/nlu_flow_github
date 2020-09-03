from tqdm import tqdm

from nlu_flow.utils import meta_db_client

import os, sys

def analyze_intent(text: str, analysis_dict: dict):
    ## 1. check intent_name is set directly
    if 'intent_' in text:
        return {'name': text.strip(), 'confidence':1.0, 'classifier':'intent_pre_analyzer'}

    ## 2. check text exist in pre_analsis_dictionary


    




