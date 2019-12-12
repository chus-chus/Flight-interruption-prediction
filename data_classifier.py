import os
import sys
import pandas as pd
import pyspark
import numpy as np
import config
import shutil
import argparse

from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
from datetime import datetime, timedelta

from pyspark.ml.classification import DecisionTreeClassifier, DecisionTreeClassificationModel
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics

csv_path = os.getcwd() + '/resources/trainingData/'

DWuser = "jesus.maria.antonanzas"
DWpass = "DB200598"

# converts to datetime.date format
def date_format(str):
    return '-'.join(("20"+str[4:6],str[2:4],str[0:2]))

def right_key(str):
    # Returns a KEY in ('aircraftid','dateid') format.
    return (str[-10:-4], date_format(str[-30:-24]))

def get_values(str):
    # Returns the sensor value of a sample.
    return float(str.split(';')[2])

def format_data_from_sources(session, fromdate):

    dw = (session.read
                .format("jdbc")
                .option("driver","org.postgresql.Driver")
                .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                .option("dbtable", "aircraftutilization")
                .option("user", DWuser)
                .option("password", DWpass)
                .load())

    # subsystem, starttime
    # aircraftregistration

    load_dw = (dw.select("aircraftid", "timeid", "flighthours",
                     "unscheduledoutofservice", "flightcycles", "delayedminutes")
                 .where("timeid >= '"+fromdate+"'")).rdd

    # For each flight, mark if there has been unscheduled maintenance sometime
    # in the next seven days.
    flights = (load_dw
             # select aircraftid, timeid, FH, FC, DM
               .map(lambda t: ((t[0], t[1].strftime('%Y-%m-%d')), (round(float(t[2]), 2), int(t[4]), int(t[5])))))

    # Get the average sensor values rdd: e.g. (('XY-SFN', '2014-12-04'), 60.624)
    averages = (sc.wholeTextFiles(csv_path+"*.csv")
                  .flatMapValues(lambda t: t.split('\n'))
                  .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
                  .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
                  .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
                  .mapValues(lambda t: t[0]/t[1]))

    # final data matrix, ((FH, FC, DM, avg(sensor), response))
    matrix = (flights.join(averages)
                      # remove key values as its better to convert to 'libsvm' format
                      # ((aircraft, date), (FH, FC, DM, avg(sensor)))
                      .map(lambda t: (t[1][0][0], t[1][0][1], t[1][0][2], t[1][1]))
                      .cache())

    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default= 3.6, help='Python compatibility (just for Alex, sorry!)', type = float)
    parser.add_argument('--fromdate', default= '2010-01-01', help='Pick flights from this date onwards: YYYY/MM/DD', type = str)

    args = parser.parse_args()

    version = args.version
    fromdate = args.fromdate

    version = 'python3.7' if (version == 3.7) else 'python3.6'

    sc = config.config_env(version)
    sess = SparkSession(sc)

    # build the data matrix
    matrix = format_data_from_sources(sess, fromdate)

    labeledpoints = matrix.map(lambda t: LabeledPoint(99, t[:3]))

    # convert matrix rdd into libsvm matrix, label is not existing so mark it as '99'
    # labeledpointscheto = (matrix.map(lambda t: LabeledPoint(99, t[:3]))
    #                 .map(lambda p: MLUtils._convert_labeled_point_to_libsvm(p)))

    # for x in labeledpointscheto.collect():
    #     print(x)

    matrix_path = os.getcwd() + '/test_matrix/'
    model_path = os.getcwd() + '/model/'

    # Remove previous version, if one
    shutil.rmtree(matrix_path, onerror = lambda f, path, exinfo: ())

    # Save matrix
    MLUtils.saveAsLibSVMFile(labeledpoints, matrix_path)

    # Load the matrix. This can change in future versions
    testdata = sess.read.format("libsvm").option("numFeatures", "3").load(matrix_path)
    print(f'Data matrix saved in {matrix_path}')

    testdata = labeledpoints.toDF("indexedLabel", "indexedFeatures")

    # for x in testdata.collect():
    #     print(x)


    # load model
    model = DecisionTreeClassificationModel.load(model_path)

    # Make predictions.
    predictions = model.transform(testdata)

    # Display results.
    predictions.show()
