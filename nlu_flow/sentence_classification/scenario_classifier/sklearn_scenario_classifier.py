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

def train_scenario_classifier():
    # load dataset
    scenario_table_list = ["intent-rules", "nlu-intent-entity-utterances"]

    utterances = []
    labels = []

    ## scenario domain
    for scenario_table in scenario_table_list:
        scenario_data = meta_db_client.get(scenario_table)
        for data in tqdm(
            scenario_data, desc=f"collecting table data : {scenario_table}"
        ):
            if type(data) != dict:
                print(f"check data type : {data}")
                continue

            try:
                utterances.append(normalize(data["utterance"]))
                labels.append(data['intent_id']['Intent_ID'])
            except:
                print (f'check data: {data}')

    ## get synonym data for data augmentation
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

    X_train, X_test, y_train, y_test = train_test_split(
        utterances, labels, random_state=88
    )

    svc = make_pipeline(CountVectorizer(analyzer="char_wb"), SVC(probability=True))
    print("scenario classifier training(with SVC)")
    svc.fit(X_train, y_train)
    print("model training done, validation reporting")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

    reportDict = {}
    for k, v in classification_report(y_test, y_pred, output_dict=True).items():
        if 'avg' in k:
            reportDict[k] = v

    with open("report.md", "w") as reportFile:
        print("scenario classification result\n", file=reportFile)
        print(reportDict, file=reportFile)

    # save scenario classifier model
    with open("scenario_classifier_model.svc", "wb") as f:
        dill.dump(svc, f)
        print("scenario_classifier model saved : scenario_classifier_model.svc")


train_scenario_classifier()
