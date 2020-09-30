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

def train_chitchat_classifier():
    # load dataset
    utterances = []
    labels = []

    response_dict = {}
    chitchat_response_dataset = meta_db_client.get('nlu-chitchat-responses')
    for data in chitchat_response_dataset:
        if data['class_name'] is None:
            continue

        if data['class_name']['classes'] not in response_dict.keys():
            response_dict[data['class_name']['classes']] = []
        response_dict[data['class_name']['classes']].append(data['response'])

    chitchat_dataset = meta_db_client.get('nlu-chitchat-utterances')
    for data in chitchat_dataset:
        if data['class_name'] is None:
            continue

        utterances.append(normalize(data['utterance']))
        labels.append(data['class_name']['classes'])

    X_train, X_test, y_train, y_test = train_test_split(utterances, labels, test_size=0.1, random_state=88)

    svc = make_pipeline(CountVectorizer(analyzer="char_wb", ngram_range=(1,3)), SVC(probability=True))
    print("chitchat classifier training(with SVC)")
    svc.fit(X_train, y_train)
    print("model training done, validation reporting")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

    with open("report.md", "w") as reportFile:
        print("chitchat classification result", file=reportFile)
        print(classification_report(y_test, y_pred, output_dict=True), file=reportFile)

    # save chitchat classifier model
    with open("chitchat_classifier_model.svc", "wb") as f:
        dill.dump(svc, f)
        print("chitchat_classifier model saved : chitchat_classifier_model.svc")

    with open('response_dict.dill', 'wb') as responseFile:
        dill.dump(response_dict, responseFile)

train_chitchat_classifier()
