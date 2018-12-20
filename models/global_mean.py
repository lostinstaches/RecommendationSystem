
import pandas as pd
import numpy as np


# A global mean class which uses the global mean to for evey prediction.

class GlobalMean_Classifier: 
    def __init__(self):
       ()
    def fit(self, train):   
        """ Calculates the global mean of the training data and uses this value to predict every rating
            input:      data         -training data to calculate the mean from
        """  
        self.prediction = train['rating'].mean()
    def predict(self, test):
        """ Create predictions for the (user, item) pairs of the test set.
            input:      test           -the test set to predict

            output:                    -the predicted ratings as a pandas dataframe
        """
        test_pred = test[['user', 'item']]
        test_pred['est'] = self.prediction
        test_pred = test_pred.sort_values(['user','item'],ascending=[True, True])
        test_pred.index = range(len(test_pred))
        return test_pred
    def predictOne(self, user, item):
        """ Create predictions for a single (user, item) pair
            input:      user           -the user associated with the predicition
                        item           -the item associated with the predicition

            output:                    -the predicted rating

        """
        return self.prediction
    def rmse(self, predictions):
        """ Calculates the RMSE of the predictions given as argument
            input:      predictions    -the predictions to include in the RMSE calculations

            output:                    -the rmse of the predictions
        """
        return np.sqrt(np.mean((predictions - self.prediction) ** 2))