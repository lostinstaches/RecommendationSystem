from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark import SQLContext
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from sklearn.model_selection import ParameterGrid
from pyspark.mllib.recommendation import Rating
import pandas as pd
import random


# A wrapper class for the pyspark ALS algorithm implmenetation

class ALS_Classifier: 
    def __init__(self, sc, numIterations=60, rank=11, regParam=0.09):
        sc.setCheckpointDir('checkpoint/')
        self.spark = SQLContext(sc)
        self.sc = sc        
        self.regParam = regParam
        self.rank = rank
        self.numIterations = numIterations
        self.evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        
    def fit(self, data):
        """ Fits the ALS model to training data
            input:      data           -training data  
                        tx          -feature matrix
                        w           -weight vector

            output:     l           -the loss when predicting the labels of the samples in tx using weight vector w 
        """
        from pyspark.mllib.recommendation import ALS
        schema = StructType([ StructField("user", IntegerType(), True)\
                       ,StructField("item", IntegerType(), True)\
                       ,StructField("rating", IntegerType(), True)])
        ratings = self.spark.createDataFrame(data, schema=schema)
        print(ratings.show())
        self.clf = ALS.train(ratings, self.rank, self.numIterations, self.regParam, seed=0)
    
    def predict(self, test):
        """ Create predictions for the (user, item) pairs of the test set.
            input:      test           -the test set to predict

            output:                    -the predicted ratings as a pandas dataframe
        """
        test = test[['user', 'item']]
        schema = StructType([ StructField("user", IntegerType(), True)\
                       ,StructField("item", IntegerType(), True)])
        final = self.spark.createDataFrame(test, schema=schema)
        test_pred = self.clf.predictAll(final.rdd.map(lambda x: (x[0], x[1])))\
            .map(lambda r: ((r[0], r[1]), r[2]))\
            .toDF()\
            .toPandas()
        test_pred['user'] = test_pred['_1'].apply(lambda x: x['_1'])
        test_pred['item'] = test_pred['_1'].apply(lambda x: x['_2'])
        test_pred['est'] = test_pred['_2']
        test_pred = test_pred.drop(['_1', '_2'], axis=1)
        test_pred = test_pred.sort_values(by=['user', 'item'])
        test_pred.index = range(len(test_pred))
        return test_pred
    def predictOne(self, user, item):
        """ Create predictions for a single (user, item) pair
            input:      user           -the user associated with the predicition
                        item           -the item associated with the predicition

            output:                    -the predicted rating

        """
        p = self.sc.parallelize([(user, item)])
        prediction = self.clf.predictAll(p)\
            .map(lambda r: ((r[0], r[1]), r[2]))\
            .toDF()\
            .toPandas()
        return prediction['_2']
    def rmse(self, predictions):
        """ Calculates the RMSE of the predictions given as argument
            input:      predictions    -the predictions to include in the RMSE calculations

            output:                    -the rmse of the predictions
        """
        return self.evaluator.evaluate(predictions)

    def grid_search(self, train, test, param_grid, outputFileName="als_grid.csv"):
        """ Performs a grid search on the ALS model using the trainset for crossvalidation and uses the test set to make final predictions.
            The reason why we use a seperate test set is because it does not appear to be possible (at least we have not found out how to do it)
            to extract the rmse score of from the pyspark crossvalidation function. Therefore we decided to use a seperate test set which we use to approximate the rmse
            when using the crossValidate function of pyspark

            Saves the output of the cross validation to the file system as a csv file.

            input:      train          -train           set to perform cross-validation on
                        train          -outputFileName  the name of the output file
        """

        schema = StructType([ StructField("user", IntegerType(), True)\
                       ,StructField("item", IntegerType(), True)\
                       ,StructField("rating", IntegerType(), True)])
        train = self.spark.createDataFrame(train,schema=schema)
        test = self.spark.createDataFrame(test,schema=schema)

        from pyspark.ml.recommendation import ALS
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

        ranks = []
        regParams = []
        rmses = []
        grid = ParameterGrid(param_grid)
        for params in grid:
            regParam = params['regParam']
            rank = params['rank']
            print("trying: {} {}".format(rank, regParam))
        
            als = ALS()
            paramGrid = ParamGridBuilder() \
                .addGrid(als.regParam, [regParam]) \
                .addGrid(als.rank, [rank]) \
                .addGrid(als.maxIter, [60]) \
                .build()
            crossval = CrossValidator(estimator=als,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=5)
            cvModel = crossval.fit(train)
            prediction = cvModel.transform(test)
            rmse = evaluator.evaluate(prediction)
            print("Root-mean-square error = " + str(rmse))
            ranks.append(rank)
            regParams.append(regParam)
            rmses.append(rmse)
        df = pd.DataFrame.from_dict({'rank': ranks, 'reg': regParams, 'rmse': rmses})
        df.to_csv(outputFileName)