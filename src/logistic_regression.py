import os
import random
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

os.chdir('..')
path = os.getcwd()

def load_data(data_file):
    return pd.read_csv(data_file, header = 'infer')

def train_test_split(df, split = 0.2):
    test = df.sample(frac = split)
    train = df.loc[~df.index.isin(test.index)]
    return train, test

def logistic_metrics(test, pred):
    test['predictions'] = pred
    test['difference'] = test['predictions'] - test['target']
#    false_pos = test[test['difference'] == 1].shape[0]
#    false_neg = test[test['difference'] == -1].shape[0]
#    true_pos = test[test['predictions'] == 1].shape[0] - false_pos
    results = confusion_matrix(test['target'], test['predictions'])
    print('Confusion Matrix :')
    print(results) 
    print('Accuracy Score :', round(accuracy_score(test['target'], test['predictions']), 4))
    print('Report :')
    print(classification_report(test['target'], test['predictions']))

def baseline_random(train, test):
    random.seed(1)
    predictions = []
    unique_classes = list(set(train['target'].values))
    for row in test.iterrows():
        predictions.append(random.choice(unique_classes))
    return predictions

def main():
    data_file = path + '\\heart_disease\\heart.csv'
    df = load_data(data_file)
    train, test = train_test_split(df, 0.25)
    pred = baseline_random(train, test)
    logistic_metrics(test, pred)

if __name__ == '__main__':
    main()