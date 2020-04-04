import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, classification_report, roc_curve,auc
import warnings
warnings.filterwarnings('ignore')

def model(algorithm, dtrain_X, dtrain_Y, dtest_X, dtest_Y, blind_X, blind_Y, algo = 0):
    cols = dtrain_X.columns
    algorithm.fit(dtrain_X[cols],dtrain_Y)
    predictions = algorithm.predict(dtest_X[cols]) # Predictions for the test dataset
    blindPredictions = algorithm.predict(blind_X[cols]) # Predictions for the blind dataset
    print (algorithm)

    print ("Accuracy score : ", round(accuracy_score(dtest_Y, predictions), 3))
    print ("Recall score   : ", round(recall_score(dtest_Y, predictions), 3))
    print ("classification report :\n", classification_report(dtest_Y, predictions))

    if(algo == 1):
        fname = "logit"
    elif(algo == 2):
        fname = "randomForest"

    s0 = dtest_Y.reset_index(level=0)
    s1 = pd.Series(predictions, name = 'predicted')
    pd.concat([s0, s1], axis = 1).to_csv(fname + "modelPred.csv", sep=',', index = False) # Save predictions to file
    print("Model predictions for test data written to " + fname +"modelPred.csv")

    s2 = blind_Y.reset_index(level=0)
    s3 = pd.Series(blindPredictions, name = 'predicted')
    pd.concat([s2.CustomerID, s3], axis = 1).to_csv(fname + "blindPred.csv", sep=',', index = False) # Save blind test predictions to file
    print("Model predictions for blind data written to "+ fname + "blindPred.csv")

    return classification_report(dtest_Y, predictions, output_dict = True)

def missingImpute(data, numTypes, objTypes):
    data_new  = data[[i for i in data.columns if i not in numTypes + objTypes]]

    ## Impute median value for features with missing numerical data
    for j, col in enumerate(numTypes):
        data_new[j] = data[col].fillna(int(data[col].median()))
        data_new = data_new.rename(columns = {j: col})

    ## Impute mode (most frequent) value for features with missing categorical data
    for k, col in enumerate(objTypes):
        data_new[k] = data[col].fillna(data[col].mode()[0])
        data_new = data_new.rename(columns={k: col})

    return data_new

def main():
    print("+--------------+")
    print("Reading the Data")
    path = os.getcwd()
    applicationFile = path + '/ds-app.tsv'
    borrowerFile = path + '/ds-borrower.tsv'
    creditFile = path + '/ds-credit.tsv'
    resultFile = path + '/ds-result.tsv'

    files = [applicationFile, borrowerFile, creditFile, resultFile]
    holder = {}
    for file in files:
        with open(file) as fp:
            reader = csv.reader(fp, delimiter='\t')
            header = next(reader, None)
            table = []
            for row in reader:
                elements = row[0].split()
                table.append(elements)
            df = pd.DataFrame(table, columns = header)
            holder[file] = df

    applicationdf = holder[applicationFile]
    borrowerdf = holder[borrowerFile]
    creditdf = holder[creditFile]
    resultdf = holder[resultFile]

    print("+-----------------+")
    print("Setting the indices")
    applicationdf.set_index('CustomerID', inplace=True)
    borrowerdf.set_index('CustomerID', inplace=True)
    creditdf.set_index('CustomerID', inplace=True)
    resultdf.set_index('CustomerID', inplace=True)

    print("+-------------------+")
    print("Merging the data sets")
    join1 = pd.merge(applicationdf,
                      borrowerdf,
                      right_index=True,
                      left_index=True,
                      how = 'left')

    join2 = pd.merge(join1,
                      creditdf,
                      right_index=True,
                      left_index=True,
                      how = 'left')

    join3 = pd.merge(join2,
                      resultdf,
                      right_index=True,
                      left_index=True,
                      how = 'left')

    join3.index = join3.index.map(int)
    join3.sort_index(inplace=True)
    join3[["LoanPayoffPeriodInMonths","RequestedAmount","InterestRate",
           "YearsInCurrentResidence","Age","NumberOfDependantsIncludingSelf",
           "CurrentOpenLoanApplications"]] = join3[["LoanPayoffPeriodInMonths","RequestedAmount","InterestRate",
                                                    "YearsInCurrentResidence","Age","NumberOfDependantsIncludingSelf",
                                                    "CurrentOpenLoanApplications"]].apply(pd.to_numeric)

    blindTestdf = join3[join3['WasTheLoanApproved'].isnull()]
    combineddf = join3[-join3['WasTheLoanApproved'].isnull()]

    print("+---------------+")
    print("Train Test split.")
    train, test = train_test_split(combineddf, test_size = 0.25)
    num_dtypes = [i for i in train.select_dtypes(include = np.number).columns if i not in applicationdf.columns]
    obj_dtypes = [i for i in train.select_dtypes(include = np.object).columns if i not in applicationdf.columns]
    obj_dtypes = [x for x in obj_dtypes if x != 'WasTheLoanApproved']

    print("+--------------------+")
    print("Handling missing data.")
    # train_new  = train[[i for i in train.columns if i not in num_dtypes + obj_dtypes]]
    # test_new  = test[[i for i in test.columns if i not in num_dtypes + obj_dtypes]]
    # blind_new = blindTestdf[[i for i in blindTestdf.columns if i not in num_dtypes + obj_dtypes]]
    train_new = missingImpute(train, num_dtypes, obj_dtypes)
    test_new = missingImpute(test, num_dtypes, obj_dtypes)
    blind_new = missingImpute(blindTestdf, num_dtypes, obj_dtypes)

    ## Impute median value for features with missing numerical data
    # for j, col in enumerate(num_dtypes):
    #     train_new[j] = train[col].fillna(int(train[col].median()))
    #     train_new = train_new.rename(columns = {j: col})
    #     test_new[j] = test[col].fillna(int(test[col].median()))
    #     test_new = test_new.rename(columns = {j: col})
        # blind_new[j] = blindTestdf[col].fillna(int(blindTestdf[col].median()))
        # blind_new = blind_new.rename(columns = {j: col})

    ## Impute mode (most frequent) value for features with missing categorical data
    # for k, col in enumerate(obj_dtypes):
    #     train_new[k] = train[col].fillna(train[col].mode()[0])
    #     train_new = train_new.rename(columns={k: col})
    #     test_new[k] = test[col].fillna(test[col].mode()[0])
    #     test_new = test_new.rename(columns={k: col})
        # blind_new[j] = blindTestdf[col].fillna(blindTestdf[col].mode()[0])
        # blind_new = blind_new.rename(columns = {j: col})

    ## Converting 'Y'/'N' values to 1/0
    blind_new["WasTheLoanApproved"].fillna(0, inplace=True)
    train_new.WasTheLoanApproved.replace(('Y', 'N'), (1, 0), inplace=True)
    test_new.WasTheLoanApproved.replace(('Y', 'N'), (1, 0), inplace=True)

    train_new[['WasTheLoanApproved']] = train_new[['WasTheLoanApproved']].apply(pd.to_numeric)
    test_new[['WasTheLoanApproved']] = test_new[['WasTheLoanApproved']].apply(pd.to_numeric)

    print("+--------------------+")
    print("Label and hot encoding")
    le = LabelEncoder()
    obj_dtypes = [i for i in train_new.select_dtypes(include = np.object).columns]
    for i in obj_dtypes:
        train_new[i] = le.fit_transform(train_new[i])
        test_new[i] = le.fit_transform(test_new[i])
        blind_new[i] = le.fit_transform(blind_new[i])
    train_new = pd.get_dummies(data=train_new,columns=obj_dtypes)
    test_new = pd.get_dummies(data=test_new,columns=obj_dtypes)
    blind_new = pd.get_dummies(data=blind_new,columns=obj_dtypes)

    ## Splitting training data into X, Y
    x_train = train_new[[i for i in train_new.columns if i not in ['WasTheLoanApproved']]]
    y_train = train_new.WasTheLoanApproved

    ## Splitting testing data into X, Y
    x_test = test_new[[i for i in test_new.columns if i not in ['WasTheLoanApproved']]]
    y_test = test_new.WasTheLoanApproved

    ## Splitting blind test data into X, Y
    x_blind = blind_new[[i for i in blind_new.columns if i not in ['WasTheLoanApproved']]]
    y_blind = blind_new.WasTheLoanApproved

    print("+---------------------------------------+")
    print("Running a Logistic Regression Classifier.")
    logit = LogisticRegression()
    logitModel = model(logit, x_train, y_train, x_test, y_test, x_blind, y_blind, algo = 1)
    # print(logitModel)

    print("+---------------------------------+")
    print("Running a Random Forest Classifier.")
    rfc = RandomForestClassifier()
    rfcModel = model(rfc, x_train, y_train, x_test, y_test, x_blind, y_blind, algo = 2)
    # print(rfcModel)

if __name__ == '__main__':
    main()
