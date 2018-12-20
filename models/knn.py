from surprise import KNNBaseline
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
import pandas as pd


# A wrapper class for the surprise KNN algorithm implmenetation

class KNN_Classifier: 
    def __init__(self, user_based=True,k=20, min_support=1, shrinkage=100):
        self.clf = KNNBaseline(k=k,
                                        sim_options={'name': 'pearson_baseline', 
                                                     'user_based': user_based,
                                                     'min_support' : min_support,
                                                     'shrinkage' : shrinkage})
    def fit(self, train):
        """ Fits the KNN model to training data
            input:      train         -training data which to fit the model to 
        """
        reader = Reader(rating_scale=(1, 5))
        train = Dataset.load_from_df(train, reader=reader)
        train = train.build_full_trainset()
        self.clf.fit(train)
    def predict(self, test):
        """ Create predictions for the (user, item) pairs of the test set.
            input:      test           -the test set to predict

            output:                    -the predicted ratings as a pandas dataframe
        """
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
        """ Create predictions for a single (user, item) pair
            input:      user           -the user associated with the predicition
                        item           -the item associated with the predicition

            output:                    -the predicted rating

        """
        return self.clf.predict(user, item)[3]
    def rmse(self, predictions):
        """ Calculates the RMSE of the predictions given as argument
            input:      predictions    -the predictions to include in the RMSE calculations

            output:                    -the rmse of the predictions
        """
        return accuracy.rmse(predictions, verbose=True)