
from helpers import load_data
from surprise import Reader
from surprise import KNNBaseline
from surprise import Dataset
from surprise.model_selection import GridSearchCV
import pandas as pd


def grid_search(data, param_grid, algo, folds):
    gs = GridSearchCV(algo, 
                  param_grid, 
                  measures=['rmse', 'mae'], 
                  return_train_measures=True,
                  cv=folds, 
                  joblib_verbose=1)
    gs.fit(data)
    return gs

def main():
    data = load_data('./data/de_data_train.csv')
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data, reader=reader)
    param_grid = {'n_epochs': [60],# 10, 15], 
              'n_factors': [40],# 10, 15, 20, 25, 30], 
              'reg_all': [0.1],# 0.01, 0.001], 
              'lr_all': [0.06, 0.08, 0.12]}#, 0.01, 0.001]}
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data, reader=reader)
    gs = grid_search(data=surprise_data, param_grid=param_grid, algo=SVD,folds=3)
    pd.DataFrame.from_dict(gs.cv_results).to_csv('knn_train.csv')

if __name__ == "__main__":
    main()

    


