from helpers import rmse
import numpy as np
import pandas as pd
from models.als import ALS_Classifier

def blend(train, val, test, models, global_mean,debug=True):

    print("Starting blend....")
    stacked_train = pd.DataFrame()
    stacked_test = pd.DataFrame()
    for i, model in enumerate(models):
        print("Training model {}".format(i))
        model.fit(train)
        pred_train = model.predict(val)
        pred_train = pred_train.rename(columns={'est': i}).reset_index()
        stacked_train = pd.concat([stacked_train, pred_train[i]], axis=1)

        """
        if debug:
            pred_test = model.predict(test)
            pred_test = pred_test.rename(columns={'est': i}).reset_index()
            print(pred_test.head())
            stacked_test = pd.concat([stacked_test,  pred_test[i]], axis=1)
         """

    val = val.sort_values(['user','item'],ascending=[True, True])
    
    if debug:
        test = test.sort_values(['user','item'],ascending=[True, True])

    # Get blending weights

    b = val['rating']
    a = stacked_train.values
    w = np.linalg.lstsq(a, b)[0]

    print("Weights:")
    print(w)

    # Another way to compute the prediction
    # final_preds = np.sum(stacked_test * w,axis=1).values 

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

    final_preds = []
    for j in range(len(preds[0])):
        p = 0
        for i in range(len(models)):
            p = p + preds[i][j] * w[i]
        final_preds.append(p)

    
    if debug:
        #final_preds = list(map(lambda x: x[2] + global_mean, preds))
        print("DEBUG - Test rmse for model {} - {}".format(i, rmse(np.clip(final_preds, 1, 5), test['rating'])))
    res = pd.DataFrame.from_dict({'rating': final_preds, 'user': test['user'], 'item': test['item']})
    return res#pd.DataFrame({'rating': preds, 'item': test['item'], 'user': test['user']})
    


"""

    preds = []
    for _, test_row in test.iterrows():
        user = test_row['user']
        item = test_row['item']
        pred = 0
        for i, model in enumerate(models):
            p = model.predictOne(user, item)
            pred = pred + p[3] * w[i]
        preds.append((user, item, pred))



"""