from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from tqdm import tqdm
from pprint import pprint

from nlu_flow.utils import meta_db_client
from nlu_flow.preprocessor.text_preprocessor import normalize

import os, sys
import dill


def train_domain_classifier():
    # load dataset
    scenario_table_list = ["intent-rules", "nlu-intent-entity-utterances"]
    faq_table_list = ["nlu-faq-questions"]
    out_of_domain_table_list = ["nlu-chitchat-utterances", "nlu-slang-utterances"]

    utterances = []
    labels = []

    total_scenario_utter_num = 0
    total_faq_utter_num = 0
    total_OOD_utter_num = 0

    ## scenario domain
    for scenario_table in scenario_table_list:
        scenario_data = meta_db_client.get(scenario_table)
        for data in tqdm(
            scenario_data, desc=f"collecting table data : {scenario_table}"
        ):
            if type(data) != dict:
                print(f"check data type : {data}")
                continue

            total_scenario_utter_num += 1

            if "data_type" in data.keys():
                if data["data_type"] == "training":
                    if "faq" in data["intent_id"]["Intent_ID"].lower():
                        utterances.append(normalize(data["utterance"]))
                        labels.append("faq")
                    elif data["intent_id"]["Intent_ID"] == "intent_OOD":
                        utterances.append(normalize(data["utterance"]))
                        labels.append("out_of_domain")
                    elif data["intent_id"]["Intent_ID"] not in ["intent_미지원"]:
                        utterances.append(normalize(data["utterance"]))
                        labels.append("scenario")

            else:
                utterances.append(normalize(data["utterance"]))
                labels.append("scenario")

    ## get synonym data for data augmentation(for FAQ domain data augmentation)
    synonyms = []
    synonym_data = meta_db_client.get("meta-entities")
    for data in tqdm(
        synonym_data, desc=f"collecting synonym data for data augmentation ..."
    ):
        if type(data) != dict:
            print(f"check data type : {data}")
            continue

        synonyms += [
            normalize(each_synonym.get("synonym"))
            for each_synonym in data.get("meta_synonyms")
        ] + [normalize(data.get("Entity_Value"))]

    ## FAQ domain
    for faq_table in faq_table_list:
        faq_data = meta_db_client.get(faq_table)
        for data in tqdm(faq_data, desc=f"collecting table data : {faq_table}"):
            target_utterance = normalize(data["question"])

            if total_faq_utter_num > max(total_scenario_utter_num, total_OOD_utter_num):
                break

            # check synonym is included
            for synonym_list in synonyms:
                for i, prev_value in enumerate(synonym_list):
                    if prev_value in target_utterance:
                        for j, post_value in enumerate(synonym_list):
                            if i == j:
                                continue
                            utterances.append(
                                target_utterance.replace(prev_value, post_value)
                            )
                            labels.append("faq")
                            total_faq_utter_num += 1
                        break

            utterances.append(target_utterance)
            labels.append("faq")
            total_faq_utter_num += 1

    ## out of domain
    for ood_table in out_of_domain_table_list:
        ood_data = meta_db_client.get(ood_table)
        for data in tqdm(ood_data, desc=f"collecting table data : {ood_table}"):
            utterances.append(normalize(data["utterance"]))
            labels.append("out_of_domain")
            total_OOD_utter_num += 1

    ### add some additional out of domain data for avoing class imbalance
    slang_training_data = meta_db_client.get("nlu-slang-trainings", max_num=total_scenario_utter_num)
    for i, data in tqdm(
        enumerate(slang_training_data),
        desc=f"collecting table data : nlu-slang-trainings ...",
    ):
        if i > total_scenario_utter_num:
            break

        if type(data) != dict:
            print (f'check data type: {data}')
            continue

        utterances.append(normalize(data["utterance"]))
        labels.append("out_of_domain")
        total_OOD_utter_num += 1

    X_train, X_test, y_train, y_test = train_test_split(
        utterances, labels, random_state=88
    )

    print ('utterance data distribution')
    print (f'scenario : {total_scenario_utter_num}')
    print (f'FAQ : {total_faq_utter_num}')
    print (f'out of domain : {total_OOD_utter_num}')

    svc = make_pipeline(CountVectorizer(analyzer="char_wb"), SVC(probability=True))
    print("domain classifier training(with SVC)")
    svc.fit(X_train, y_train)
    print("model training done, validation reporting")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

    with open("report.md", "w") as reportFile:
        print("domain classification result", file=reportFile)
        print(classification_report(y_test, y_pred), file=reportFile)

    # save domain classifier model
    with open("domain_classifier_model.svc", "wb") as f:
        dill.dump(svc, f)
        print("domain_classifier model saved : domain_classifier_model.svc")


train_domain_classifier()
