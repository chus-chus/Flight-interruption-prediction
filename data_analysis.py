import os
import sys
import pyspark
import config
import shutil

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

def trainModel(data, sc):
    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Let us undersample the majority class

    trainingrdd = trainingData.select('*').rdd

    yesdata = trainingrdd.filter(lambda t: t[0] == 1.0)
    nodata = trainingrdd.filter(lambda t: t[0] == 0.0)

    # We sample a bit more 0's than 1's
    sampleRatio = (float(yesdata.count())/float(trainingrdd.count()))*1.7
    sampled_nodata = nodata.sample(False, sampleRatio)
    trainingrdd = yesdata.union(sampled_nodata)

    print(f'\n trainingRDD with {trainingrdd.count()} rows, [yes: {yesdata.count()}, no: {sampled_nodata.count()}]')

    trainingData = trainingrdd.toDF()

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    print("Model trained")

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Create performance matrix
    accuracy = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

    recall = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")

    acc = accuracy.evaluate(predictions)
    print(f'  Accuracy: {acc}')
    print(f'Test error: {1 - acc}')
    print(f'    Recall: {recall.evaluate(predictions)}')

    treeModel = model.stages[2]

    return treeModel


if __name__ == "__main__":

    # Python compatibility (just for Alex, sorry!)
    version = "python3.7" if (len(sys.argv) == 2 and sys.argv[1] == 'a') else "python3.6"

    sc = config.config_env(version)

    sess = SparkSession(sc)

    matrix_path = os.getcwd() + '/data_matrix/'
    model_path = os.getcwd() + '/model/'

    # load data matrix
    matrix = sess.read.format("libsvm").option("numFeatures", "3").load(matrix_path)

    # train the model
    model = trainModel(matrix, sc)

    # remove previous model version, if one
    shutil.rmtree(model_path, onerror = lambda f, path, exinfo: ())

    # save it
    model.save(model_path)

    print(f'Model saved in {model_path}')
