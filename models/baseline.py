from surprise import BaselineOnly
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
import pandas as pd

class Baseline_Classifer: 
    def __init__(self):
        self.clf = BaselineOnly()
    def fit(self, train):
        reader = Reader(rating_scale=(1, 5))
        train = Dataset.load_from_df(train, reader=reader)
        train = train.build_full_trainset()
        self.clf.fit(train)
    def predict(self, test):
        reader = Reader(rating_scale=(1, 5))
        test = Dataset.load_from_df(test, reader=reader)
        test = test.build_full_trainset().build_testset()
        test_pred = self.clf.test(test)
        test_pred = pd.DataFrame(test_pred)
        test_pred = test_pred.rename(columns={'uid':'user', 'iid': 'item'})
        test_pred = test_pred.sort_values(['user','item'],ascending=[True, True])
        test_pred.index = range(len(test_pred))
        return test_pred
    def predictOne(self, user, item):
        return self.clf.predict(user, item)[3]
    def rmse(self, predictions):
        return accuracy.rmse(predictions, verbose=True)