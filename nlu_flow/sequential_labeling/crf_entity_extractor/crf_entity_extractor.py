from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from tqdm import tqdm
from pprint import pprint

from nlu_flow.utils import meta_db_client


from utils import (
    tokenize,
    tokenize_fn,
    convert_ner_data_format,
    word2features,
    sent2features,
    sent2labels,
    sent2tokens,
    bio_classification_report,
)

import re
import pycrfsuite


def train_crf_entity_extractor():
    entity_dataset = []

    ## get synonym data for data augmentation
    synonyms = []
    synonym_data = meta_db_client.get("meta-entities")
    for data in tqdm(
        synonym_data, desc=f"collecting synonym data for data augmentation ..."
    ):
        if type(data) != dict:
            print(f"check data type : {data}")
            continue

        synonyms.append(
            [each_synonym.get("synonym") for each_synonym in data.get("meta_synonyms")]
            + [data.get("Entity_Value")]
        )

    scenario_data = meta_db_client.get("nlu-intent-entity-utterances")

    for data in tqdm(scenario_data, desc=f"generating entity dataset ..."):
        if data["entities"] is not None and len(data["entities"]) > 0:
            entity_dataset.append(
                convert_ner_data_format(data["utterance"], data["entities"])
            )

    # split entity dataset by train & val
    X_train, X_test, y_train, y_test = train_test_split(
        [sent2features(s) for s in entity_dataset],
        [sent2labels(s) for s in entity_dataset],
        random_state=88,
        test_size=0.1,
    )

    trainer = pycrfsuite.Trainer(verbose=False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params(
        {
            "c1": 1.0,  # coefficient for L1 penalty
            "c2": 1e-3,  # coefficient for L2 penalty
            "max_iterations": 50,  # stop earlier
            # include transitions that are possible, but not observed
            "feature.possible_transitions": True,
            # minimum frequency
            #'feature.minfreq': 5
        }
    )

    print("entity model training(with CRF)")
    trainer.train("crf_entity_extractor.crfsuite")
    print("entity model saved : crf_entity_extractor.crfsuite")

    print("entity model train done, validation reporting")
    tagger = pycrfsuite.Tagger()
    tagger.open("crf_entity_extractor.crfsuite")

    y_pred = []
    for test_feature in X_test:
        y_pred.append(tagger.tag(test_feature))

    print(bio_classification_report(y_test, y_pred))

    with open("report.md", "a+") as report_file:
        report = bio_classification_report(y_test, y_pred, output_dict=True)
        print("\nEntity Classification Performance", file=report_file)
        pprint(f"micro avg: {report['micro avg']}", stream=report_file)
        pprint(f"macro avg: {report['macro avg']}", stream=report_file)
        pprint(f"weighted avg: {report['weighted avg']}", stream=report_file)
        pprint(f"samples avg: {report['samples avg']}", stream=report_file)


train_crf_entity_extractor()
