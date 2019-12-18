"""
Data analysis pipe
@author: Jesus Antonanzas Acero, Alex Carrillo Alza
@version: "1.0"
@email: "jesus.maria.antonanzas@est.fib.upc.edu, alex.carrillo.alza@est.fib.upc.edu"
@info: BDA, GCED, Big Data Analytics project
@date: 16/12/2019

Description
-----------
Given a data matrix in the specified 'libsvm' format and generated with 'data_management.py'
trains a Decision Tree classifier with it, undersampling the majority class so
that new percentages are approx. (60% '0', 40% '1'). Then, computes different
metrics and saves the model locally.

Usage
-----------
Run after 'data_management.py' or after training matrix is saved in local.
In the arguments, one should specify if using Python version 3.6 or 3.7.
This script saves a trained Decision Tree model in the current path ('model').

Steps enforced
-----------
1. Configure Spark environment
2. Adjust model training tools
3. Split data into training and test sets
4. Balance training data
5. Build the model
6. Fit the model
7. Compute performance metrics on test data
8. Save the trained model locally
"""
import os
import sys
import pyspark
import config
import argparse
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
    # index labels, adding metadata to the label column.
    # fit on whole dataset to include all labels in index.
    # refer to step 2
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    # refer to step 2
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

    # Split the data into training and test sets (30% held out for testing).
    # Refer to step 3
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Balance training data. Refer to step 4
    # Let us undersample the majority class '0'.
    trainingrdd = trainingData.select('*').rdd

    yesdata = trainingrdd.filter(lambda t: t[0] == 1.0)
    nodata = trainingrdd.filter(lambda t: t[0] == 0.0)

    # We sample a bit more 0's than 1's. Note that because the explanatory variables
    # we are using do not explain much of the response the error when
    # performing undersampling 50/50 is very high: close to 50%. If we increase the
    # number of "no" in the training data, though, error will be low, but at
    # the cost of not predicting "yes" at all (very low recall).
    sampleRatio = (float(yesdata.count())/float(trainingrdd.count()))*1.7
    sampled_nodata = nodata.sample(False, sampleRatio)
    trainingrdd = yesdata.union(sampled_nodata)

    print(f'\n trainingRDD with {trainingrdd.count()} rows, [yes: {yesdata.count()}, no: {sampled_nodata.count()}]')

    trainingData = trainingrdd.toDF()

    # Create the model. Refer to step 5
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline. Refer to step 5
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model. This also runs the indexers. Refer to step 6
    model = pipeline.fit(trainingData)

    print("Model trained")

    # Make predictions on test data. Refer to step 7
    predictions = model.transform(testData)
    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(5)

    # Create performance matrix. Refer to step 7
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default= 3.6, help='Python compatibility (just for Alex, sorry!)', type = float)

    args = parser.parse_args()

    version = args.version

    # Python compatibility (just for Alex, sorry!)
    version = "python3.7" if (len(sys.argv) == 2 and sys.argv[1] == 'a') else "python3.6"

    # Build Spark envirenoment. Refer to step 1
    sc = config.config_env(version)

    sess = SparkSession(sc)

    # data matrix loading path and model saving path
    matrix_path = os.getcwd() + '/data_matrix/'
    model_path = os.getcwd() + '/model/'

    # load data matrix
    matrix = sess.read.format("libsvm").option("numFeatures", "3").load(matrix_path)

    # Model training pipeline. Refer to steps 2 to 7
    model = trainModel(matrix, sc)

    # remove previous model version, if one
    shutil.rmtree(model_path, onerror = lambda f, path, exinfo: ())

    # save it. Refer to step 8
    model.save(model_path)

    print(f'Model saved in {model_path}')
