import os
import random
import numpy as np
import pandas as pd

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
#    true_pos = test[test['difference'] == 0].shape[0]
    false_pos = test[test['difference'] == 1].shape[0]
    false_neg = test[test['difference'] == -1].shape[0]
    true_pos = test[test['predictions'] == 1].shape[0] - false_pos
    precision = round(true_pos/(true_pos + false_pos), 4)
    recall = round(true_pos/(true_pos + false_neg), 4)
    print("Precision :", precision)
    print("Recall :", recall)

def baseline_random(train, test):
    predictions = []
    unique_classes = list(set(train['target'].values))
#    print(unique_classes)
    for row in test.iterrows():
        predictions.append(random.choice(unique_classes))
    return predictions

def main():
    data_file = path + '\\heart_disease\\heart.csv'
    df = load_data(data_file)
    train, test = train_test_split(df, 0.25)
    pred = baseline_random(train, test)
    logistic_metrics(test, pred)
    return pred

if __name__ == '__main__':
    yhat = main()