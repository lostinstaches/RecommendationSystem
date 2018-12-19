# EPFL Recommeder system

Our best performing model is an ensemble model and yielded a RMSE of 1.021 on the CrowdAI test set.

### Run the following command to generate the predictions which yielded our highest CrowdAi score:

`python run.py`

Produce a submission file for the EPFL CrowdAi Challange:

`python run.py`

The run.py script does the following:

1. Creates out ensemble model
2. Trains each model on the training data
3. Finds the optimal weights for combining the predictions.
4. Makes predictions on the CrowdAI test 
5. Create a submission file

### Perform grid search

In the `grid_search` directory there are files for launching grid searches for the SVD, KNN and ALS models. Simple run `python als_grid.py` to start grid_searching the ALS
model for instance.

## Dependencies

The following external libraries are used in this project

1. PySpark (http://spark.apache.org/docs/2.2.0/api/python/pyspark.html)
2. FindSpark (https://github.com/minrk/findspark)
3. Surprise (http://surpriselib.com/)
4. Pandas (https://pandas.pydata.org/)
5. Numpy (http://www.numpy.org/)
6. Scikit-learn (https://scikit-learn.org/stable/)