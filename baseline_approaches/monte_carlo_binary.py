import random
import time
from sklearn.metrics import classification_report

## Need to split our test data into X and Y => x_test (to get the IDs for whom we are predicting)
## and y_test (to test our random guesses with the ground truth)

start = time.time()
precision = 0
recall = 0
f1 = 0
for i in range(10000):
    randomGuess = [random.randint(0, 1) for _ in range(x_test.shape[0])] ## We make a binary/bool prediction for all IDs in our test set
    report = classification_report(y_test, randomGuess, output_dict = True) ## Check against actuals/ground truth
    precision += report['weighted avg']['precision']
    recall += report['weighted avg']['recall']
    f1 += report['weighted avg']['f1-score']
stop = time.time()
print("Precision Average :", round(precision/10000.0, 4))
print("Recall Average :", round(recall/10000.0, 4))
print("f1-score Average :", round(f1/10000.0, 4))
print("Time Elapsed :", round(stop - start, 3), "seconds.")
