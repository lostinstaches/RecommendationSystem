from pyspark.sql.types import IntegerType
from pyspark.sql.types import StructType
from pyspark.sql.types import StructField
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark import SQLContext
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from sklearn.model_selection import ParameterGrid
import pandas as pd
import random
#from pyspark.mllib.recommendation import ALS


class ALS_Classifier: 
    def __init__(self, sc):
        sc.setCheckpointDir('checkpoint/')
        self.spark = SQLContext(sc)
        self.sc = sc        
        #self.rank = 8
        #self.regParam = 0.06
        #self.numIterations = 60
        #self.regParam = 0.06
        self.numIterations = 60
        self.regParam = 0.09
        self.rank = 11
        self.evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

        """
        self.clf = ALS(rank=8, 
              maxIter=50, 
              regParam=0.01,
              userCol="user", 
              itemCol="item", 
              ratingCol="rating",
              checkpointInterval=10, 
              intermediateStorageLevel="MEMORY_AND_DISK", 
              finalStorageLevel="MEMORY_AND_DISK")
        """
        
        
    def fit(self, data):
        from pyspark.mllib.recommendation import Rating
        from pyspark.mllib.recommendation import ALS
        schema = StructType([ StructField("user", IntegerType(), True)\
                       ,StructField("item", IntegerType(), True)\
                       ,StructField("rating", IntegerType(), True)])
        ratings = self.spark.createDataFrame(data, schema=schema)
        print(ratings.show())
        self.clf = ALS.train(ratings, self.rank, self.numIterations, self.regParam, seed=0)

        """
        schema = StructType([ StructField("user", IntegerType(), True)\
                       ,StructField("item", IntegerType(), True)\
                       ,StructField("rating", IntegerType(), True)])
        ratings = self.spark.createDataFrame(data, schema=schema)
        self.model = self.clf.fit(ratings)
        """
        
    def predict(self, test):
        
        test = test[['user', 'item']]
        schema = StructType([ StructField("user", IntegerType(), True)\
                       ,StructField("item", IntegerType(), True)])
        final = self.spark.createDataFrame(test, schema=schema)
        test_pred = self.clf.predictAll(final.rdd.map(lambda x: (x[0], x[1])))\
            .map(lambda r: ((r[0], r[1]), r[2]))\
            .toDF()\
            .toPandas()
        print("SHAPE INSIDE")
        print(test_pred.shape)
        test_pred['user'] = test_pred['_1'].apply(lambda x: x['_1'])
        test_pred['item'] = test_pred['_1'].apply(lambda x: x['_2'])
        test_pred['est'] = test_pred['_2']
        test_pred = test_pred.drop(['_1', '_2'], axis=1)
        test_pred = test_pred.sort_values(by=['user', 'item'])
        test_pred.index = range(len(test_pred))
        """
        schema = StructType([ StructField("user", IntegerType(), True)\
                       ,StructField("item", IntegerType(), True)\
                       ,StructField("rating", IntegerType(), True)])
        ratings = self.spark.createDataFrame(test,schema=schema)
        p = self.model.transform(ratings).toPandas()
        test_pred = p.rename(columns={'user': 'user', 'item': 'item', 'rating': 'r_ui', 'prediction': 'est'})
        test_pred = test_pred.sort_values(['user','item'],ascending=[True, True])
        """
        return test_pred
    def predictOne(self, user, item):
        p = self.sc.parallelize([(user, item)])
        prediction = self.clf.predictAll(p)\
            .map(lambda r: ((r[0], r[1]), r[2]))\
            .toDF()\
            .toPandas()
        return prediction['_2']
    def crossValidate_normal(self, data):
        self.model = self.clf.fit(data, seed=10)
    def predict_normal(self, test):
        p = self.model.transform(test)
        return p
    def rmse(self, predictions):
        return self.evaluator.evaluate(predictions)
    def predict_value(self, user, item):
        return 

    def grid_search(self, train, test):

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
        param_grid = {'rank': [11],
                      'regParam' : [0.07,0.08,0.09,0.1,0.11,0.12]}
        grid = ParameterGrid(param_grid)
        #testdata = test.rdd.map(lambda p: (p[0], p[1]))
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
                        numFolds=3)
            cvModel = crossval.fit(train)
            prediction = cvModel.transform(test)
            rmse = evaluator.evaluate(prediction)
            print("Root-mean-square error = " + str(rmse))
            ranks.append(rank)
            regParams.append(regParam)
            rmses.append(rmse)
        df = pd.DataFrame.from_dict({'rank': ranks, 'reg': regParams, 'rmse': rmses})
        df.to_csv('als_grid_4_reg.csv')