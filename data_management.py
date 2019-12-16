import os
import sys
import pyspark
import numpy as np
import config
import shutil
import argparse

from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from datetime import datetime, timedelta

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

def createGenerator(date):
    for i in range(7): yield (date - timedelta(i), 1)

def create_priordays(date):
    return createGenerator(date)

def response(value):
    return 0 if value is None else 1


def format_data_from_sources(sc):

    session = SparkSession(sc)

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

    # subsystem, starttime
    # aircraftregistration

    load_dw = dw.select("aircraftid", "timeid", "flighthours",
                     "unscheduledoutofservice", "flightcycles", "delayedminutes").rdd

    load_amos = (amos.select("aircraftregistration", "starttime", "subsystem")
                     .where("subsystem = '3453' and (kind = 'Delay' or kind = 'AircraftOnGround' or kind == 'Safety')").rdd)

    # we pick the right sensor from AMOS database and generate artifical dates,
    # which we use to add the response variable
    maintenance = (load_amos
                   .map(lambda t: (t[0], create_priordays(t[1].date())))
                   # generate 6 dates before each unshceduled maintenance. INCLOIM EL DIA QUE MIREM??????????
                   .flatMapValues(lambda t: t)
                   # aircraftid, date, response
                   .map(lambda t: ((t[0], t[1][0]), t[1][1]))
                   # delete duplicates
                   .reduceByKey(lambda t1,t2: t1))

    # For each flight, mark if there has been unscheduled maintenance sometime
    # in the next seven days.
    ACuti_Mevents = (load_dw
                     # select aircraftid, timeid, unscheduledoutofservice, FH, FC, DM
                     .map(lambda t: ((t[0], t[1]), (round(t[3]), round(float(t[2]), 2), int(t[4]), int(t[5]))))
                     # We join with the sensors.
                     .leftOuterJoin(maintenance)
                     # t[1][0][0] (unscheduledoutofservice) !!!!!!!
                     .map(lambda t: ((t[0][0], t[0][1].strftime('%Y-%m-%d')), (t[1][0][1], t[1][0][2], t[1][0][3], response(t[1][1])))))

    return ACuti_Mevents


def data_from_csvs(sc, sess, loadfrom, csv_path):

    # Get csvs from hdfs, remember to deploy avro dependencies "org.apache.spark:spark-avro_2.11:2.4.3"
    # for this option previous execution of "load_into_hdfs.py" is required.
    if loadfrom == "hdfs":
        averages = (sess.read.format("avro")
                              .load('hdfs://localhost:9000/user/chusantonanzas/sensordata')
                              .rdd
                              .map(lambda t: ((t[0][0], t[0][1]), t[1])))

    # read and process csvs from local
    elif loadfrom == "local":
        averages = (sc.wholeTextFiles(csv_path+"*.csv")
                      .flatMapValues(lambda t: t.split('\n'))
                      .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
                      .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
                      .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
                      .mapValues(lambda t: t[0]/t[1]))

    return averages

def join_csvs_dwinfo(sc, averages, ACuti_Mevents):

    # final data matrix, ((FH, FC, DM, avg(sensor), response))
    matrix = (ACuti_Mevents.join(averages)
                           # remove key values as its better to convert to 'libsvm' format
                           .map(lambda t: (t[1][0][0], t[1][0][1], t[1][0][2], t[1][1], t[1][0][3]))
                           # ((aircraft, date), (FH, FC, DM, avg(sensor), response))
                           # .mapValues(lambda t: (t[0][0], t[0][1], t[0][2], t[1], t[0][3]))
                           .cache())
    return matrix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default= 3.6, help='Python compatibility (just for Alex, sorry!)', type = float)
    parser.add_argument('--loadfrom', default= 'local', help='Sensor data (csv) load method: "hdfs", "local".', type = str)
    parser.add_argument('--csvpath', default= os.getcwd() + '/resources/trainingData/', help='CSV path for "local" option', type = str)

    args = parser.parse_args()

    version = args.version
    loadfrom = args.loadfrom
    csv_path = args.csvpath

    # configure env. variables
    sc = config.config_env(version)
    sess = SparkSession(sc)

    # read from amos and acuti necessary metrics
    ACuti_Mevents = format_data_from_sources(sc)

    # read sensors info from hdfs
    averages = data_from_csvs(sc, sess, loadfrom, csv_path)

    # create final data matrix
    matrix = join_csvs_dwinfo(sc, averages, ACuti_Mevents)

    # convert matrix rdd into libsvm matrix
    labeledpoints = matrix.map(lambda t: LabeledPoint(t[4], t[:3]))

    matrix_path = os.getcwd() + '/data_matrix/'

    # remove previous matrix version, if one
    shutil.rmtree(matrix_path, onerror = lambda f, path, exinfo: ())

    # save matrix
    MLUtils.saveAsLibSVMFile(labeledpoints, matrix_path)
    print(f'Data matrix saved in {matrix_path}')
