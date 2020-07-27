import os
import h2o
import numpy as np
import pandas as pd
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

os.chdir('..')

DATA_PATH = os.getcwd() + '\\'
fname = 'heart.csv'

def logistic_runner(x_cols, y_col, train, test, validation):
    pass

def main():
    h2o.init(max_mem_size_GB=16)
    heart_data = pd.read_csv(DATA_PATH + fname)
    heart_df = h2o.H2OFrame(heart_data)
    print(heart_df.col_names)
    y = 'target'
    x = heart_df.col_names
    x.remove(y)
    print("Response = " + y)
    print("Predictors = " + str(x))
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal', 'target']
    for col in cat_cols:
        heart_df[col] = heart_df[col].asfactor()
    print(heart_df['target'].levels())
    train, test, validation = heart_df.split_frame(ratios=[.8, .1])
    print(heart_df.shape)
    print(train.shape)
    print(test.shape)
    print(validation.shape)
    log_reg = H2OGeneralizedLinearEstimator(family = "binomial", alpha = 1)
    log_reg.train(x=x, y= y, training_frame=train, validation_frame=validation, model_id="glm_logistic_regression")
#    print(log_reg.__dict__)
    print(log_reg.confusion_matrix())
    print(log_reg.coef())
#    print(log_reg.predict(test_data=test))
#    print(log_reg.model_performance(test_data=test).rmse())
#    log_reg.std_coef_plot()
    log_reg_performance = log_reg.model_performance(test)
    print(log_reg_performance)

if __name__ == '__main__':
    main()