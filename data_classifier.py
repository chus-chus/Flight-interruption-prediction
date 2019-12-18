"""
(new) data classification pipe
@author: Jesus Antonanzas Acero, Alex Carrillo Alza
@version: "1.0"
@email: "jesus.maria.antonanzas@est.fib.upc.edu, alex.carrillo.alza@est.fib.upc.edu"
@info: BDA, GCED, Big Data Analytics project
@date: 16/12/2019

Description
-----------
Gets new flights (all parting from a specific date) from 'aircraftutilization',
gets their corresponing average sensor values and predicts their response variable
(if an unscheduled maintenance event is to happen sometime in the next 7 days) given
a previously trained Decision Tree classifier. Saves results locally.

Usage
-----------
Run after 'data_analysis.py', where the Decision Tree Classifier was trained and
saved. Can specify python version (3.7 or 3.6), date from where to begin picking
new flights from 'aircraftutilization' and, if loading sensor data from local or
HDFS and if the first case is true, the csv's path (specify if not in /resources/trainingData/).
See arguments help for more details.

Steps enforced
-----------
1. Configure Spark environment
2. Read and process (that is, create response variable) aircraft utilization
   information data from relational databases
3. Read sensor data from local (in this case compute averages for each aircraft and date)
   or HDFS (in this case averages are already processed) and format it correctly
4. Enrich aircraft utilization information with sensor data (join observations)
5. Transform resulting data set into 'libsvm' format for prediction
6. Load Decission Tree classifier
7. Predict response variable
8. Save results
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

# Converts a string like "yy-dd-mm" to format "yyyy-mm-dd". Refer to step 3.
def date_format(str):
    return '-'.join(("20"+str[4:6],str[2:4],str[0:2]))

# Returns a key in format ('aircraftid','dateid') from /resources/csvs filenames
# Refer to step 3.
def right_key(str):
    return (str[-10:-4], date_format(str[-30:-24]))

# Gets the sensor value of an observations in the sensor data csv's. Refer to step 3.
def get_values(str):
    return float(str.split(';')[2])

# Reads and format aircraft utilization metrics from PostgreSQL corresponding to flights
def format_data_from_sources(sc, sess, fromdate):

    # open reading pipe
    dw = (sess.read
              .format("jdbc")
              .option("driver","org.postgresql.Driver")
              .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
              .option("dbtable", "aircraftutilization")
              .option("user", DWuser)
              .option("password", DWpass)
              .load())

    # load variables of interest
    load_dw = (dw.select("aircraftid", "timeid", "flighthours",
                     "unscheduledoutofservice", "flightcycles", "delayedminutes")
                 .where("timeid >= '"+fromdate+"'")).rdd

    # select aircraftid, timeid, FH, FC, DM
    flights = (load_dw.map(lambda t: ((t[0], t[1].strftime('%Y-%m-%d')), (round(float(t[2]), 2), int(t[4]), int(t[5])))))

    return flights

# Load and process sensor data from local CSV's or HDFS. Refer to step 3.
def data_from_csvs(sc, sess, loadfrom, csv_path):

    # get sensordata from HDFS. Remember to deploy avro dependencies for Spark Version > 2.4
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

# Joins aircraft utilization metrics with sensor data and formats it.
# Refer to step 4.
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
    # returns a few observations: from '2016-09-07'

    args = parser.parse_args()

    loadfrom = args.loadfrom
    csv_path = args.csvpath
    version = args.version
    fromdate = args.fromdate

    version = "python3.6" if version == 3.6 else "python3.7"

    # configure env. variables. Refer to step 1
    sc = config.config_env(version)
    sess = SparkSession(sc)

    # read from 'AMOS' and 'aircraftutilization' necessary metrics.
    # Refer to step 2
    flights = format_data_from_sources(sc, sess, fromdate)

    # read sensor information from HDFS or local csv's. Refer to step 3
    averages = data_from_csvs(sc, sess, loadfrom, csv_path)

    # Step 4

    # create enriched aircraft utilization metrics (join sensor data).
    matrix = join_csvs_dwinfo(sc, averages, flights)

    # format previous rdd to 'labeled points'. As response is not existing,
    # mark it as '99'
    labeledpoints = matrix.map(lambda t: LabeledPoint(99, t[1][:3]))

    # get (local) saving path for the test data and loading path for the model
    matrix_path = os.getcwd() + '/test_matrix/'
    model_path = os.getcwd() + '/model/'

    # Remove previous test matrix version, if one
    shutil.rmtree(matrix_path, onerror = lambda f, path, exinfo: ())

    # Save matrix. This can change in future versions so that there is no
    # need to save and load it.
    MLUtils.saveAsLibSVMFile(labeledpoints, matrix_path)
    print(f'Test data matrix saved in {matrix_path}')

    # load the matrix.
    testdata = (sess.read.format("libsvm")
                .option("numFeatures", "3")
                .load(matrix_path)
                .toDF("indexedLabel", "indexedFeatures"))

    # load model. Refer to step 6
    model = DecisionTreeClassificationModel.load(model_path)

    # make predictions. Refer to step 7
    predictions = model.transform(testdata)

    # results saving path
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
    # Refer to step 8
    predictions.rdd.map(lambda t: ((t[4]),(t[1]))).saveAsTextFile(results_path)
