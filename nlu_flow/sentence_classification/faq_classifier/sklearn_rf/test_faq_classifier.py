from sklearn.metrics import classification_report
from tqdm import tqdm

from nlu_flow.preprocessor.text_preprocessor import normalize

import dill

model = None
response_dict = None
with open('./faq_classifier_model.rf', 'rb') as f:
    model = dill.load(f)
    print ('faq_classifier_model load success')
    print (model.classes_)

#prepare test dataset
with open('golden_set.tsv','r') as dataFile:
    lines = dataFile.readlines()[1:]

    X_test = []
    y_test = []

    for line in tqdm(lines, desc='preparinge test dataset ...'):
        if len(line.split('\t')) < 3:
            continue

        X_test.append(normalize(line.split('\t')[1]))
        y_test.append(line.split('\t')[0])

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))