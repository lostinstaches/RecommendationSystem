
import pandas as pd
import numpy as np
class GlobalMean_Classifier: 
    def __init__(self):
       ()
    def fit(self, train):     
        self.prediction = train['rating'].mean()
    def predict(self, test):
        test_pred = test[['user', 'item']]
        test_pred['est'] = self.prediction
        test_pred = test_pred.sort_values(['user','item'],ascending=[True, True])
        test_pred.index = range(len(test_pred))
        return test_pred
    def predictOne(self, user, item):
        return self.prediction
    def rmse(self, predictions):
        return np.sqrt(np.mean((predictions - self.prediction) ** 2))