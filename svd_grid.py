
from helpers import load_data
from surprise import Reader
from surprise import SVD
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
    param_grid = {'n_epochs': [30]
              'n_factors': [30]
              'reg_all': [0.1],
              'lr_all': [0.01]}
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data, reader=reader)
    gs = grid_search(data=surprise_data, param_grid=param_grid, algo=SVD,folds=5)
    print(gs.best_score['rmse'])    
    #pd.DataFrame.from_dict(gs.cv_results).to_csv('svd_train_v8.csv')

if __name__ == "__main__":
    main()

    


