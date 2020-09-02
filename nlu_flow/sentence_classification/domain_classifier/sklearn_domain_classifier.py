from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from tqdm import tqdm
from pprint import pprint

import os, sys
import dill

sys.path.append(
    os.path.dirname(
        os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    )
)

from utils import meta_db_client
from preprocessor.text_preprocessor import preprocess

def train_domain_classifier():
    # load dataset
    scenario_table_list=['intent-rules', 'nlu-intent-entity-utterances']
    faq_table_list=['nlu-faq-questions']
    out_of_domain_table_list=['nlu-chitchat-utterances','nlu-slang-utterances']

    utterances = []
    labels = []
    
    ## scenario domain
    for scenario_table in scenario_table_list:
        scenario_data = meta_db_client.get(scenario_table)
        for data in tqdm(scenario_data, desc=f'collecting table data : {scenario_table}'):
            if 'data_type' in data.keys():
                if data['data_type'] == 'training':
                    if 'faq' in data['intent_id']['Intent_ID'].lower():
                        utterances.append(preprocess(data['utterance']))
                        labels.append('faq')
                    elif data['intent_id']['Intent_ID'] == 'intent_OOD':
                        utterances.append(preprocess(data['utterance']))
                        labels.append('out_of_domain')
                    elif data['intent_id']['Intent_ID'] not in ['intent_미지원']:
                        utterances.append(preprocess(data['utterance']))
                        labels.append('scenario')
 
            else:
                utterances.append(preprocess(data['utterance']))
                labels.append('scenario')
 
    ## FAQ domain
    for faq_table in faq_table_list:
        faq_data = meta_db_client.get(faq_table)
        for data in tqdm(faq_data, desc=f'collecting table data : {faq_table}'):
            utterances.append(preprocess(data['question']))
            labels.append('faq')

    ## out of domain
    for ood_table in out_of_domain_table_list:
        ood_data = meta_db_client.get(ood_table)
        for data in tqdm(ood_data, desc=f'collecting table data : {ood_table}'):
            utterances.append(preprocess(data['utterance']))
            labels.append('out_of_domain')

    X_train, X_test, y_train, y_test = train_test_split(utterances, labels, random_state=88)

    svc = make_pipeline(CountVectorizer(analyzer="char_wb"), SVC(probability=True))
    print("domain classifier training(with SVC)")
    svc.fit(X_train, y_train)
    print("model training done, validation reporting")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test, y_pred))

    with open('report.md', 'w') as reportFile:
        print(classification_report(y_test, y_pred), file=reportFile)

    #save domain classifier model
    with open('domain_classifier_model.svc','wb') as f:
        dill.dump(svc, f)
        print ('domain_classifier model saved : domain_classifier_model.svc')

train_domain_classifier()
