import findspark
findspark.init()
from helpers import load_data
from pyspark import SparkContext
from models.als import ALS_Classifier
from sklearn.model_selection import train_test_split

sc = SparkContext("local", "App Name")

# Launches a grid search on the ALS model

def main():
    data = load_data('./data/de_data_train.csv')
    als = ALS_Classifier(sc)
    train, test = train_test_split(data, test_size=0.2)
    
    param_grid = {'rank': [11],
                      'regParam' : [0.07,0.08,0.09,0.1,0.11,0.12]}

    als.grid_search(train, test, param_grid)

if __name__ == "__main__":
    main()

    