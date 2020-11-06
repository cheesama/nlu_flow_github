from nlu_flow.utils import meta_db_client
from tqdm import tqdm

from datetime import datetime

import dill
import os, sys
import random
import json

# load dataset
scenario_table_list = ["intent-rules", "nlu-intent-entity-utterances"]

## get synonym data for data augmentation
synonyms = {}
synonym_data = meta_db_client.get("meta-entities")
for data in tqdm(
    synonym_data, desc=f"collecting synonym data for data augmentation ..."
):
    if type(data) != dict:
        print(f"check data type : {data}")
        continue

    synonyms[data["Entity_Type"]] = [data["Entity_Value"]]
    if "meta_synonyms" in data and len(data["meta_synonyms"]) > 0:
        for synonym in data["meta_synonyms"]:
            synonyms[data["Entity_Type"]].append(synonym["synonym"])

utterances = []
labels = []
test_utterances = []
test_labels = []

## scenario dataset preparation
for scenario_table in scenario_table_list:
    scenario_data = meta_db_client.get(scenario_table)
    for data in tqdm(scenario_data, desc=f"collecting table data : {scenario_table}"):
        if type(data) != dict:
            print(f"check data type : {data}")
            continue

        if "data_type" in data.keys():
            if data["data_type"] == "training":
                utterances.append(data["utterance"])
                labels.append(data["intent_id"]["Intent_ID"])
            elif data["data_type"] == "goldenset":
                test_utterances.append(data["utterance"])
                test_labels.append(data["intent_id"]["Intent_ID"])

            # synonym augmentation
            if "entities" not in data.keys() or data["entities"] is None:
                continue

            if len(data["entities"]) == 1:  # synonym augmentation
                entity_type = data["entities"][0]["entity"]
                entity_value = data["entities"][0]["value"]

                if entity_type in synonyms.keys():
                    for synonym in synonyms[entity_type]:
                        if "data_type" in data.keys():
                            if data["data_type"] == "training":
                                print(data["utterance"].replace(entity_value, synonym))

                                utterances.append(
                                    data["utterance"].replace(entity_value, synonym)
                                )
                                labels.append(data["intent_id"]["Intent_ID"])
                            elif data["data_type"] == "goldenset":
                                test_utterances.append(
                                    data["utterance"].replace(entity_value, synonym)
                                )
                                test_labels.append(data["intent_id"]["Intent_ID"])
        else:
            utterances.append(data["utterance"])
            labels.append(data["intent_id"]["Intent_ID"])

with open("scenario_dataset.tsv", "w") as scenarioData:
    scenarioData.write("text\tclass\n")

    for i, utter in enumerate(utterances):
        scenarioData.write(utter.strip().replace("\t", " "))
        scenarioData.write("\t")
        scenarioData.write(labels[i].strip().replace("\t", " "))
        scenarioData.write("\n")

with open("scenario_test_dataset.tsv", "w") as scenarioData:
    scenarioData.write("text\tclass\n")

    for i, utter in enumerate(test_utterances):
        scenarioData.write(utter.strip().replace("\t", " "))
        scenarioData.write("\t")
        scenarioData.write(test_labels[i].strip().replace("\t", " "))
        scenarioData.write("\n")

os.system("rm -rf results")
os.system("ludwig experiment --dataset scenario_dataset.tsv --config_file config.yml")
os.system(
    "ludwig evaluate --dataset scenario_test_dataset.tsv --model_path results/experiment_run/model"
)

# write result to file
with open("results/experiment_run/test_statistics.json") as f:
    test_result = json.load(f)

    meta_db_client.post(
        "nlu-model-reports",
        {
            "name": "ludwig_scenario_classifier",
            "version": datetime.today().strftime("%Y-%m-%d_%H:%M:%S"),
            "report": test_result,
        },
    )

    with open("report.md", "w") as reportFile:
        reportFile.write("scenario classifier(transformer based) test result\n")
        json.dump(test_result["class"]["overall_stats"], reportFile, indent=4)
