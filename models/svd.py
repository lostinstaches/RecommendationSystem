from surprise import KNNBaseline
from surprise import Reader
from surprise import Dataset
import pandas as pd
from surprise import accuracy
from surprise import SVD

# A wrapper class for the surprise SVD algorithm implmenetation

class SVD_Classifier: 
    def __init__(self, n_factors=60, lr_all=0.01, reg_all=0.1, n_epochs=100):
        self.clf = SVD(n_factors=n_factors,lr_all=lr_all, reg_all=reg_all, n_epochs=n_epochs, random_state=0)
    def fit(self, train):
        """ Fits the SVD model to training data
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