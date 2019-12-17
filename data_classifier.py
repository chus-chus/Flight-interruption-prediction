"""
(new) data classification pipe
@author: Jesus Antonanzas Acero, Alex Carrillo Alza
@version: "1.0"
@email: "jesus.maria.antonanzas@est.fib.upc.edu, alex.carrillo.alza@est.fib.upc.edu"
@info: BDA, GCED, Big Data Analytics project
@date: 16/12/2019

Usage
-----------

Description
-----------

Steps enforced
-----------
1-
"""
import os
import sys
import pyspark
import config
import shutil
import argparse

from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassificationModel

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

def format_data_from_sources(sc, sess):

    dw = (sess.read
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

    # select aircraftid, timeid, FH, FC, DM
    flights = (load_dw.map(lambda t: ((t[0], t[1].strftime('%Y-%m-%d')), (round(float(t[2]), 2), int(t[4]), int(t[5])))))

    return flights


def data_from_csvs(sc, sess, loadfrom, csv_path):

    # Get sensordata from hdfs, remember to deploy avro dependencies for Spark Version > 2.4
    # "org.apache.spark:spark-avro_2.11:2.4.3"
    # for this option previous execution of "load_into_hdfs.py" is required.
    if loadfrom == "hdfs":
        averages = (sess.read.format("avro")
                             .load('hdfs://localhost:9000/user/chusantonanzas/sensordata')
                             .rdd
                             .map(lambda t: ((t[0][0], t[0][1]), t[1])))

    # read and process csv's sensordata from local
    elif loadfrom == "local":
        averages = (sc.wholeTextFiles(csv_path+"*.csv")
                      .flatMapValues(lambda t: t.split('\n'))
                      .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
                      .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
                      .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
                      .mapValues(lambda t: t[0]/t[1]))

    return averages

# Final data matrix, (FH, FC, DM, avg(sensor)), (aircraftid, date)
def join_csvs_dwinfo(sc, averages, flights):
    matrix = (flights.join(averages)
                     .map(lambda t: ((t[0]),(t[1][0][0], t[1][0][1], t[1][0][2], t[1][1])))
                     .cache())
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default= 3.6, help='Python compatibility (just for Alex, sorry!)', type = float)
    parser.add_argument('--fromdate', default= '2016-09-07', help='Pick flights from this date onwards: YYYY/MM/DD', type = str)
    parser.add_argument('--loadfrom', default= 'local', help='Sensor data (csv) load method: "hdfs", "local".', type = str)
    parser.add_argument('--csvpath', default= os.getcwd() + '/resources/trainingData/', help='CSV path for "local" option', type = str)
    # returns a few observations: 2016-09-07

    args = parser.parse_args()

    loadfrom = args.loadfrom
    csv_path = args.csvpath
    version = args.version
    fromdate = args.fromdate

    version = "python3.6" if version == 3.6 else "python3.7"

    sc = config.config_env(version)
    sess = SparkSession(sc)

    # read from amos and acuti necessary metrics
    flights = format_data_from_sources(sc, sess)

    # read sensors info from hdfs
    averages = data_from_csvs(sc, sess, loadfrom, csv_path)

    # create final data matrix
    matrix = join_csvs_dwinfo(sc, averages, flights)

    # convert matrix rdd into libsvm matrix, label is not existing so mark it as '99'
    labeledpoints = matrix.map(lambda t: LabeledPoint(99, t[1][:3]))

    matrix_path = os.getcwd() + '/test_matrix/'
    model_path = os.getcwd() + '/model/'

    # Remove previous version, if one
    shutil.rmtree(matrix_path, onerror = lambda f, path, exinfo: ())

    # Save matrix
    MLUtils.saveAsLibSVMFile(labeledpoints, matrix_path)
    print(f'Data matrix saved in {matrix_path}')

    # Load the matrix. This can change in future versions so that there is no
    # need to save and load it.
    testdata = (sess.read.format("libsvm")
                .option("numFeatures", "3")
                .load(matrix_path)
                .toDF("indexedLabel", "indexedFeatures"))

    # Load model.
    model = DecisionTreeClassificationModel.load(model_path)

    # Make predictions.
    predictions = model.transform(testdata)

    # Let's now save results in a text file
    results_path = os.getcwd() + '/prediction_results/'

    # Remove previous results, if ones
    shutil.rmtree(results_path, onerror = lambda f, path, exinfo: ())

    # Display results. For larger dataframes please pipe the output to a textfile.
    print('Prediction complete for the following observations:')
    for x in matrix.collect():
        print(f'Aircraft {x[0][0]} on date {x[0][1]}')
    print('\n With the following results:')
    predictions.select("prediction", "indexedFeatures").show(predictions.count(), False)

    # Save predictions. It contains ((FH, FC, DM), prediction). Prediction = 1: there
    # will be an unscheduled maintenance event in the next 7 days.
    predictions.rdd.map(lambda t: ((t[4]),(t[1]))).saveAsTextFile(results_path)
