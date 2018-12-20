import findspark
findspark.init()
from pyspark import SparkContext
sc = SparkContext("local", "App Name")

import numpy as np 
import pandas as pd 
from helpers import load_data
from sklearn.model_selection import train_test_split
from models.knn import KNN_Classifier
from models.svd import SVD_Classifier
from models.als import ALS_Classifier
from models.global_mean import GlobalMean_Classifier
from models.baseline import Baseline_Classifer

from blend import blend
from helpers import get_submission_rows
from helpers import rmse
import sys
import random


def createSubmission(predictions):
    out = open("submission.csv","w")
    out.write('Id,Prediction\n')
    for _,prediction in predictions.iterrows():
        rating = prediction['rating']
        user = prediction['user']
        item = prediction['item']
        rating = int(np.rint(rating))
        rating = 5 if rating == 6 else rating
        rating = 1 if rating == 0 else rating
        p_string = "r{}_c{},{}\n".format(int(item), int(user), rating)
        out.write(p_string)        
    out.close()

def main():

    # [0.70542062 0.25247412 0.03072762 0.02667471]
    # [0.70408259 0.25219167 0.03046293 0.02853859]
    # [0.70408259 0.25219167 0.03046293 0.02853859]

    # Load dataset
    print("Loading dataset...")
    data = load_data('./data/de_data_train.csv')

    # Load modes
    models = [
        ALS_Classifier(sc, numIterations=60, rank=11, regParam=0.09), 
        KNN_Classifier(user_based=True, k=20, min_support=1, shrinkage=100), 
        KNN_Classifier(user_based=False, k=20, min_support=1, shrinkage=100),
        SVD_Classifier(n_factors=60, lr_all=0.01,reg_all=0.1)
    ]

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Splits dataset into three pieces train, val and test.
        tmp, test = train_test_split(data, test_size=0.2, random_state=0)
        train, val = train_test_split(tmp, test_size=0.1, random_state=0)
        predictions = blend(train, val, test, models, debug=True)
        test = test.sort_values(['user','item'],ascending=[True, True])
        print("Test rmse: {}".format(rmse(np.clip(np.rint(predictions['rating']), 1, 5), test['rating'])))
    else:
        # Splits dataset into 2 pieces train and val. Uses the submission_rows as testset 
        train, val = train_test_split(data, test_size=0.1, random_state=0)
        test = get_submission_rows()
        predictions = blend(train, val, test, models, debug=False)
        createSubmission(predictions)

if __name__ == "__main__":
    main()