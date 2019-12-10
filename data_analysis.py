import os
import pyspark
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

path = os.getcwd() + '/data_matrix/'

def trainModel(matrix, sc):
    sess = SparkSession(sc)
    # convert matrix rdd into libsvm matrix

    # labeledpoints = matrix.map(lambda t: LabeledPoint(t[4], t[:3]))
    # MLUtils.saveAsLibSVMFile(labeledpoints, path)

    data = sess.read.format("libsvm").option("numFeatures", "3").load(path)

    # Index labels, adding metadata to the label column.
    # Fit on whole dataset to include all labels in index.
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=5).fit(data)

    # Split the data into training and test sets (40% held out for testing)
    (trainingData, testData) = data.randomSplit([0.7, 0.3])

    # Let us undersample the majority class

    trainingrdd = trainingData.select('*').rdd

    yesdata = trainingrdd.filter(lambda t: t[0] == 1.0)
    nodata = trainingrdd.filter(lambda t: t[0] == 0.0)
    print(nodata.count())
    sampleRatio = float(yesdata.count()) / float(trainingrdd.count())
    sampled_nodata = nodata.sample(False, sampleRatio)
    trainingrdd = yesdata.union(sampled_nodata)

    print(trainingrdd.count(), "yes: ", yesdata.count(), "no: ", sampled_nodata.count())

    trainingData = trainingrdd.toDF()

    # Train a DecisionTree model.
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")

    # Chain indexers and tree in a Pipeline
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # Train model.  This also runs the indexers.
    model = pipeline.fit(trainingData)

    # Make predictions.
    predictions = model.transform(testData)

    # Select example rows to display.
    predictions.select("prediction", "indexedLabel", "features").show(20)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")

    recall = evaluator.evaluate(predictions)
    print(recall)
    #print("Test Error = %g " % (1.0 - accuracy))

    treeModel = model.stages[2]
    # summary only
    print(treeModel)

    return treeModel
