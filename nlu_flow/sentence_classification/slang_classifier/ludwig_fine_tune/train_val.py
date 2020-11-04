from nlu_flow.utils import meta_db_client

from tqdm import tqdm
from datetime import datetime

import dill
import os, sys
import random
import json

import tensorflow as tf
tf.debugging.set_log_device_placement(True)

utterances = []
labels = []

## faq domain
faq_data = meta_db_client.get("nlu-faq-questions")
for data in tqdm(faq_data, desc=f"collecting faq data ... "):
    if data["faq_intent"] is None or len(data["faq_intent"]) < 2:
        print(f"check data! : {data}")
        continue

    target_utterance = data["question"]

    # check synonym is included
    for synonym_list in synonyms:
        for i, prev_value in enumerate(synonym_list):
            if prev_value in target_utterance:
                for j, post_value in enumerate(synonym_list):
                    if i == j:
                        continue
                    utterances.append(target_utterance.replace(prev_value, post_value))
                    labels.append("false")

## scenario domain
scenario_data = meta_db_client.get("nlu-intent-entity-utterances")
for data in tqdm(scenario_data, desc=f"collecting table data : nlu-intent-entity-utterances"):
    if type(data) != dict:
        print(f"check data type : {data}")
        continue

    utterances.append(data["utterance"])
    labels.append("false")

## out of domain
slang_training_data = meta_db_client.get("nlu-slang-trainings")
for i, data in tqdm(enumerate(slang_training_data), desc=f"collecting table data : nlu-slang-trainings ...",):
    if type(data) != dict:
        print (f'check data type: {data}')
        continue

    utterances.append(data["utterance"])
    if data['label'] == '0':
        labels.append("false")
    else:
        labels.append("true")

slang_utterance_data = meta_db_client.get("nlu-slang-utterances")
for i, data in tqdm(enumerate(slang_utterance_data), desc=f"collecting table data : nlu-slang-utterances ...",):
    if type(data) != dict:
        print (f'check data type: {data}')
        continue

    utterances.append(data["utterance"])
    labels.append("true")

chitchat_data = meta_db_client.get("nlu-chitchat-utterances")
for data in tqdm(chitchat_data, desc=f"collecting table data : nlu-chitchat-utterances"):
    utterances.append(data["utterance"])
    labels.append("false")

with open('slang_dataset.tsv', 'w') as slangData:
    slangData.write('text\tclass\n')
    
    for i, utter in enumerate(utterances):
        slangData.write(utter.strip().replace('\t',' '))
        slangData.write('\t')
        slangData.write(labels[i].strip().replace('\t', ' '))
        slangData.write('\n')

os.system('rm -rf results')
os.system('ludwig experiment --dataset slang_dataset.tsv --config_file config.yml')

#write result to file
with open('results/experiment_run/test_statistics.json') as f:
    test_result = json.load(f)

    meta_db_client.post('nlu-model-reports', {'name': 'ludwig_slang_classifier', 'version':datetime.today().strftime("%Y-%m-%d_%H:%M:%S"), 'report': test_result})

    with open('report.md', 'w') as reportFile:
        reportFile.write('slang classification test result\n')
        json.dump(test_result['class']['overall_stats'], reportFile, indent=4)
