# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import pandas as pd

def load_data(path):
    """ Loads the ratings data specified by the path argument return the data as a pandas dataframe
        input:      path           -the path where the data is located

        output:                    -the loaded data as a pandas dataframe
    """
    data_df = pd.read_csv(path)
    data_df['user'] = data_df['Id'].str.split('_').str[1].apply(lambda x: int(x[1:]))
    data_df['item'] = data_df['Id'].str.split('_').str[0].apply(lambda x: int(x[1:]))
    data_df = data_df.rename(columns={'Prediction':'rating'})
    data_df = data_df[['user','item','rating']]
    return data_df

def rmse(predictions, label):
    """ Returns the RMSE of the predictions and the true labels
        input:      predictions           -the predictions made by some model
                    label                 -the true values of the predictions

        output:                           -the calculated rmse
    """
    return np.sqrt(np.mean((predictions - label) ** 2))

def get_submission_rows():
    """ Returns submission (user, item) pairs as a pandas dataframe
        output:                           -the user and item submission pairs
    """
    users = []
    items = []
    with open('./data/submission_rows') as samples:
        for i, sample in enumerate(samples):
            if i == 0:
                continue
            tmp = sample.split('_')
            item = int(tmp[0][1:].strip())
            user = int(tmp[1][1:].strip())
            users.append(user)
            items.append(item)
    return pd.DataFrame.from_dict({'user': users, 'item': items})


def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
    """split the ratings to training data and test data.
    Args:
        min_num_ratings: 
            all users and items we keep must have at least min_num_ratings per user and per item. 
    """
    # set seed
    np.random.seed(988)
    
    # select user and item based on the condition.
    valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
    valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
    valid_ratings = ratings[valid_items, :][: , valid_users]  
     
    rows, cols = valid_ratings.shape
    
    test_idx = np.random.rand(rows, cols) > p_test
    train = valid_ratings.copy()
    test = valid_ratings.copy()
    train[~test_idx] = 0
    test[test_idx] = 0
        
    print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
    print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
    print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
    return valid_ratings, train, test, valid_users, valid_items


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""
    num_items, num_users = train.shape
    # Kaiming He Weight Initialization
    user_features = np.random.normal(scale=np.sqrt(2/num_features), size=(num_features, num_users))
    item_features = np.random.normal(scale=np.sqrt(2/num_features), size=(num_features, num_items))
    return user_features, item_features


def create_submission(item_features, user_features, user_bias, item_bias, global_bias):
    out = open("submission.csv","w")
    out.write('Id,Prediction\n')
    with open('./data/submission_rows') as samples:
        for n, sample in enumerate(samples):
            if n == 0:
                continue
            tmp = sample.split('_')
            row = int(tmp[0][1:].strip())
            col = int(tmp[1][1:].strip())
            user_feature = user_features[:, col - 1]
            item_feature = item_features[:, row - 1]
            base = global_bias + item_bias[row - 1] + user_bias[col - 1]
            prediction = int(np.rint(base + user_feature.dot(item_feature)))
            prediction = 5 if prediction == 6 else prediction
            p_string = "r{}_c{},{}\n".format(row, col, prediction)
            out.write(p_string)        
    out.close()
    
def create_submission_baseline_estimate(user_bias, item_bias, global_bias):
    out = open("submission.csv","w")
    out.write('Id,Prediction\n')
    with open('./data/submission_rows') as samples:
        for n, sample in enumerate(samples):
            if n == 0:
                continue
            tmp = sample.split('_')
            row = int(tmp[0][1:].strip())
            col = int(tmp[1][1:].strip())
            base = global_bias + item_bias[row - 1] + user_bias[col - 1]
            prediction = int(np.rint(base))
            prediction = 5 if prediction == 6 else prediction
            p_string = "r{}_c{},{}\n".format(row, col, prediction)
            out.write(p_string)        
    out.close()
