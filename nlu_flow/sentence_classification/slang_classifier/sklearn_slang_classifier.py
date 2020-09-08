from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

from tqdm import tqdm
from pprint import pprint

from nlu_flow.utils import meta_db_client
from nlu_flow.preprocessor.text_preprocessor import normalize

import os, sys
import dill


def train_slang_classifier():
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    # load dataset
    slang_data = meta_db_client.get("nlu-slang-trainings")
    for data in tqdm(slang_data, desc=f"collecting slang data..."):
        if data["data_type"] == "train":
            X_train.append(normalize(data["utterance"]))
            y_train.append(data["label"])
        elif data["data_type"] == "val":
            X_test.append(normalize(data["utterance"]))
            y_test.append(data["label"])

    #svc = make_pipeline(CountVectorizer(analyzer="char_wb"), OneVsRestClassifier(SVC(kernel="linear", probability=True)))
    rf = make_pipeline(CountVectorizer(analyzer="char_wb"), RandomForestClassifier())

    #print("slang classifier training(with SVC)")
    print("slang classifier training(with RandomForest)")
    #svc.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    print("model training done, validation reporting")
    #y_pred = svc.predict(X_test)
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    with open("report.md", "w") as reportFile:
        print('slang classification result', file=reportFile)
        print(classification_report(y_test, y_pred), file=reportFile)

    # save slang classifier model
    '''
    with open("slang_classifier_model.svc", "wb") as f:
        dill.dump(svc, f)
        print("slang_classifier model saved : slang_classifier_model.svc")
    '''
    with open("slang_classifier_model.rf", "wb") as f:
        dill.dump(rf, f)
        print("slang_classifier model saved : slang_classifier_model.rf")

train_slang_classifier()
