"""
Data management pipe script
@author: Jesus Antonanzas Acero, Alex Carrillo Alza
@version: "1.0"
@email: "jesus.maria.antonanzas@est.fib.upc.edu, alex.carrillo.alza@est.fib.upc.edu"
@info: BDA, GCED, Big Data Analytics project
@date: 16/12/2019

Usage
-----------
If reading sensor data from HDFS, this script is to be executed 2nd, after "load_into_hdfs.py".
In the arguments, one should specify if using Python version 3.6 or 3.7, if loading sensor data
from local or HDFS and if the first case is true, the csv's path
(specify if not in /resources/trainingData/). See arguments help for more details.
The results from this script is a file "data_matrix" containing a 'libsvm'
set of files representing the data matrix to feed into the model.

Description
-----------
Extracts necesary data from sources (sensor data -locally or HDFS- and aircraft
utilization), creates the response variable to train the model with and saves
the data matrix locally.

Steps enforced
-----------
1. Configure Spark environment
2. Read and process (that is, create response variable) aircraft utilization
   information data from relational databases
3. Read sensor data from local (in this case compute averages for each aircraft and date)
   or HDFS (in this case averages are already processed) and format it correctly
4. Generate response variable
5. Enrich aircraft utilization information with sensor data (join observations)
6. Transform resulting data set into 'libsvm' format for training the model
   and save it locally
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
from pyspark.sql import SparkSession
from datetime import datetime, timedelta

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

# Auxiliary function in the creation of the response variable. Yields 6 days
# previous to "date", including "date". Refer to step 4.
def createGenerator(date):
    for i in range(7): yield (date - timedelta(i), 1)

# Calls previous generator function, creating 6 artificial dates previous to "date".
# Refer to step 4.
def create_priordays(date):
    return createGenerator(date)

# Auxiliary function in the labeling part of the response variable. Refer to step 4.
def response(value):
    return 0 if value is None else 1

# Reads and format aircraft utilization metrics from PostgreSQL corresponding to flights
# and non-scheduled maintenance events associated with sensor '3453'. Refer to step 2.
def format_data_from_sources(sc):

    session = SparkSession(sc)

    # open reading pipes
    dw = (session.read
                 .format("jdbc")
                 .option("driver","org.postgresql.Driver")
                 .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/DW?sslmode=require")
                 .option("dbtable", "aircraftutilization")
                 .option("user", DWuser)
                 .option("password", DWpass)
                 .load())

    amos = (session.read
                .format("jdbc")
                .option("driver","org.postgresql.Driver")
                .option("url", "jdbc:postgresql://postgresfib.fib.upc.edu:6433/AMOS?sslmode=require")
                .option("dbtable", "maintenanceevents")
                .option("user", DWuser)
                .option("password", DWpass)
                .load())

    # load variables of interest
    load_dw = dw.select("aircraftid", "timeid", "flighthours",
                     "unscheduledoutofservice", "flightcycles", "delayedminutes").rdd

    load_amos = (amos.select("aircraftregistration", "starttime", "subsystem")
                     .where("subsystem = '3453' and (kind = 'Delay' or kind = 'AircraftOnGround' or kind == 'Safety')").rdd)

    # pick the right sensor from AMOS database and generate artifical dates,
    # which we use to add the response variable
    maintenance = (load_amos
                   .map(lambda t: (t[0], create_priordays(t[1].date())))
                   # generate 6 dates before each unshceduled maintenance. INCLOIM EL DIA QUE MIREM??????????
                   .flatMapValues(lambda t: t)
                   # aircraftid, date, response
                   .map(lambda t: ((t[0], t[1][0]), t[1][1]))
                   # delete duplicates
                   .reduceByKey(lambda t1,t2: t1))

    # for each flight, mark if there has been unscheduled maintenance sometime
    # in the next seven days.
    ACuti_Mevents = (load_dw
                     # select aircraftid, timeid, unscheduledoutofservice, FH, FC, DM
                     .map(lambda t: ((t[0], t[1]), (round(t[3]), round(float(t[2]), 2), int(t[4]), int(t[5]))))
                     # join with the sensor information.
                     .leftOuterJoin(maintenance)
                     # (aircraftid, date), (FH, FC, DM, response)
                     .map(lambda t: ((t[0][0], t[0][1].strftime('%Y-%m-%d')), (t[1][0][1], t[1][0][2], t[1][0][3], response(t[1][1])))))

    return ACuti_Mevents

# Load and process sensor data from local CSV's or HDFS. Refer to step 3.
def data_from_csvs(sc, sess, loadfrom, csv_path):
    # get csvs from hdfs, remember to deploy avro dependencies
    # "org.apache.spark:spark-avro_2.11:2.4.3".
    # For this option previous execution of "load_into_hdfs.py" is required.
    if loadfrom == "hdfs":
        averages = (sess.read.format("avro")
                             .load('hdfs://localhost:9000/user/chusantonanzas/sensordata')
                             .rdd
                             .map(lambda t: ((t[0][0], t[0][1]), t[1])))

    # read and process csvs from local
    elif loadfrom == "local":
        averages = (sc.wholeTextFiles(csv_path+"*.csv")
                      .flatMapValues(lambda t: t.split('\n'))
                      # delete title and empty observations
                      .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
                      # compute averages for each 'date' and 'aircraftid'
                      .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
                      .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
                      .mapValues(lambda t: t[0]/t[1]))

    return averages

# Joins aircraft utilization metrics with sensor data and formats it.
# Refer to step 5.
def join_csvs_dwinfo(sc, averages, ACuti_Mevents):
    matrix = (ACuti_Mevents.join(averages)
                           # remove key values as its better to convert to 'libsvm' format
                           # ((aircraft, date), (FH, FC, DM, avg(sensor), response))
                           .map(lambda t: (t[1][0][0], t[1][0][1], t[1][0][2], t[1][1], t[1][0][3]))
                           .cache())
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default= 3.6, \
                        help='Python compatibility (just for Alex, sorry!)', type = float)
    parser.add_argument('--loadfrom', default= 'local', \
                        help='Sensor data (csv) load method: "hdfs", "local".', type = str)
    parser.add_argument('--csvpath', default= os.getcwd() + '/resources/trainingData/', \
                        help='CSV path for "local" option', type = str)

    args = parser.parse_args()

    version = args.version
    loadfrom = args.loadfrom
    csv_path = args.csvpath

    version = 'python3.6' if version == 3.6 else 'python3.7'

    # configure env. variables. Refer to step 1
    sc = config.config_env(version)
    sess = SparkSession(sc)

    # read from 'AMOS' and 'aircraftutilization' necessary metrics and create
    # response variable. Refer to steps 2 and 4
    ACuti_Mevents = format_data_from_sources(sc)

    # read sensors info from hdfs or local csv's. Refer to step 3
    averages = data_from_csvs(sc, sess, loadfrom, csv_path)

    # create enriched aircraft utilization metrics (join sensor data).
    # Refer to step 3
    matrix = join_csvs_dwinfo(sc, averages, ACuti_Mevents)

    # format previous rdd to 'labeled points'
    labeledpoints = matrix.map(lambda t: LabeledPoint(t[4], t[:3]))

    # get (local) saving path
    matrix_path = os.getcwd() + '/data_matrix/'

    # remove previous matrix version, if one
    shutil.rmtree(matrix_path, onerror = lambda f, path, exinfo: ())

    # save matrix
    MLUtils.saveAsLibSVMFile(labeledpoints, matrix_path)
    print(f'Data matrix saved in {matrix_path}')
