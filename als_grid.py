import findspark
findspark.init()
from helpers import load_data
from pyspark import SparkContext

from models.als import ALS_Classifier
from sklearn.model_selection import train_test_split

sc = SparkContext("local", "App Name")

def main():
    data = load_data('./data/de_data_train.csv')
    als = ALS_Classifier(sc)
    train, test = train_test_split(data, test_size=0.2)
    als.grid_search(train, test)

if __name__ == "__main__":
    main()

    