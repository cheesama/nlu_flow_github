from nlu_flow.utils import meta_db_client
from tqdm import tqdm

import dill
import os, sys
import random
import json

utterances = []
labels = []

## get synonym data for data augmentation(for FAQ data augmentation)
synonyms = []
synonym_data = meta_db_client.get("meta-entities")
for data in tqdm(
    synonym_data, desc=f"collecting synonym data for data augmentation ..."
):
    if type(data) != dict:
        print(f"check data type : {data}")
        continue

    synonyms.append([each_synonym.get("synonym") for each_synonym in data.get("meta_synonyms")] + [data.get("Entity_Value")])

faq_response_dict = {}
faq_data = meta_db_client.get("nlu-faq-questions")
for data in tqdm(faq_data, desc=f"collecting faq data ... "):
    if data["faq_intent"] is None or len(data["faq_intent"]) < 2:
        print(f"check data! : {data}")
        continue

    target_utterance = data["question"]
    faq_response_dict[data['faq_intent']] = data['prompt_id']

    # check synonym is included
    for synonym_list in synonyms:
        for i, prev_value in enumerate(synonym_list):
            if prev_value in target_utterance:
                for j, post_value in enumerate(synonym_list):
                    if i == j:
                        continue
                    utterances.append(target_utterance.replace(prev_value, post_value))
                    labels.append(data["faq_intent"])


with open('faq_dataset.tsv', 'w') as faqData:
    faqData.write('text\tclass\n')
    
    for i, utter in enumerate(utterances):
        faqData.write(utter.strip())
        faqData.write('\t')
        faqData.write(labels[i])
        faqData.write('\n')

with open('faq_response_dict.dill', 'wb') as responseFile:
    dill.dump(faq_response_dict, responseFile)

'''
scenario_data = meta_db_client.get("nlu-intent-entity-utterances")
for data in tqdm(random.choices(scenario_data, k=len(self.utterances)), desc=f"collecting scenario data ... "):
    self.utterances.append(data["utterance"])
    self.labels.append('scenario')
'''

os.system('rm -rf results')
os.system('ludwig experiment --dataset faq_dataset.tsv --config_file config.yml')
#os.system('ludwig evaluate --dataset golden_set.tsv --model_path results/experiment_run/model/')

#write result to file
with open('results/experiment_run/test_statistics.json') as f:
    test_result = json.load(f)

    with open('report.md', 'w') as reportFile:
        json.dump(test_result['class']['overall_stats'], reportFile, indent=4)
