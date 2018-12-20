from helpers import rmse
import numpy as np
import pandas as pd
from models.als import ALS_Classifier

def blend(train, val, test, models,debug=True):
    """ Trains a blended model on the trainset, extracts the weights for each model using the validation set
        and uses the test set to make final predicitons.
            input:      train           -the train data set
                        val             -the validation data set
                        test            -the test data set
                        models          -the base models of the blend
                        debug           -debug flag, enables some additional printing



            output:                    -the predicted ratings on the test set as a pandas dataframe
    """

    print("Starting blend....")
    stacked_train = pd.DataFrame()
    for i, model in enumerate(models):
        print("Training model {}".format(i))
        # Fit each model to the train set
        model.fit(train)
        # Evaluate model on the validation set
        pred_train = model.predict(val)
        pred_train = pred_train.rename(columns={'est': i}).reset_index()
        stacked_train = pd.concat([stacked_train, pred_train[i]], axis=1)

    val = val.sort_values(['user','item'],ascending=[True, True])
    
    if debug:
        test = test.sort_values(['user','item'],ascending=[True, True])

    # Get blending weights
    b = val['rating']
    a = stacked_train.values
    w = np.linalg.lstsq(a, b)[0]
    print("Weights:")

    # Save weights to file
    weight_df = pd.DataFrame({'weight': w})
    weight_df.to_csv('weights.csv')


    # Calculate predictions on the test set
    preds = []
    for n, model in enumerate(models):
        print("Predicting for model {}".format(n))
        predictions = []
        if isinstance(model, ALS_Classifier):
            print(test.shape)
            predictions = model.predict(test)['est']
            print(predictions.shape)
            predictions = predictions.tolist()
            preds.append(predictions)
        else:
            if debug:
                predictions = model.predict(test)['est']
                preds.append(predictions.tolist())
            else:    
                for _, test_row in test.iterrows():
                    user = test_row['user']
                    item = test_row['item']
                    predictions.append(model.predictOne(user, item))
                preds.append(predictions)

    # Combine predictions from base models and scale by their correspnding weights.
    final_preds = []
    for j in range(len(preds[0])):
        p = 0
        for i in range(len(models)):
            p = p + preds[i][j] * w[i]
        final_preds.append(p)

    
    if debug:
        print("DEBUG - Test rmse for model {} - {}".format(i, rmse(np.clip(final_preds, 1, 5), test['rating'])))

    res = pd.DataFrame.from_dict({'rating': final_preds, 'user': test['user'], 'item': test['item']})
    return res
    