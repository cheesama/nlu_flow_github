from tqdm import tqdm
from fastapi import FastAPI
from itertools import product

from nlu_flow.utils import meta_db_client

import os, sys
import random
import re

app = FastAPI()
is_ready = False

patterns = {}

## collect single pattern
single_patterns = meta_db_client.get("regexes")
for pattern in single_patterns:
    patterns[pattern['entity']] = pattern['pattern']

## collect combination pattern
combination_patterns = meta_db_client.get("regex-combinations")
for combination in combination_patterns:
    gen_regex_comb = []
    for _, pattern_list in combination['combination'].items():
        gen_regex = []
        for each_pattern in pattern_list:
            gen_regex.append(patterns[each_pattern].split('|'))
        gen_regex_comb += ['\\s'.join(regex) for regex in list(product(*gen_regex))]

    patterns[combination['entity']] = '|'.join(gen_regex_comb)

print (f'total patterns: {len(patterns)}')
#print (patterns['Stay_Period'])

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
async def match_rule_entities(text: str):
    extracted = []
    for k, v in patterns.items():
        matches = re.finditer(pattern=v, string=text)
        for match in matches:
            entity = {"start": match.start(), "end": match.end(), "value": match.group(), "confidence": 1.0, "entity": k, "extractor": "rule-entity-extractor"}
            extracted.append(entity)

    #organize overlapped entities
    remove_overlapped = []
    extracted.sort(key=lambda x: len(x.get('value')), reverse=True)
    for i, target_entity in enumerate(extracted):
        for j, compare_entity in enumerate(extracted):
            if i == j: continue
            if compare_entity['start'] <= target_entity['start'] and target_entity['end'] <= compare_entity['end']:
                break
            if j == len(extracted) - 1:
                remove_overlapped.append(target_entity)

    return remove_overlapped


