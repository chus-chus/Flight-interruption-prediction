"""
Optional HDFS loading script
@author: Jesus Antonanzas Acero, Alex Carrillo Alza
@version: "1.0"
@email: "jesus.maria.antonanzas@est.fib.upc.edu, alex.carrillo.alza@est.fib.upc.edu"
@info: BDA, GCED, Big Data Analytics project
@date: 16/12/2019

Usage
-----------
This file should be executed first, before any pipes.
It is required to deploy avro dependencies for Spark versions >= 2.4 before
executing this script:

"org.apache.spark:spark-avro_2.11:2.4.3"

Description
-----------
Load and process sensor data from CSVs and load them in HDFS.

Steps enforced
-----------
1- Configure Spark environment
2- Read CSVs, and compute sensor averages for date and aircraftid
3- Store the resulting Spark Data Frame -with columns "[aircraftid, date], [avg(sensor)]"- in HDFS
   (path can be specified via an argument)
"""

import pyspark
import sys
import config
import os
import argparse

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession


# Converts to datetime.date format. Refer to step 2.
def date_format(str):
    return '-'.join(("20"+str[4:6],str[2:4],str[0:2]))

# Returns a KEY in ('aircraftid','dateid') format. Refer to step 2.
def right_key(str):
    return (str[-10:-4], date_format(str[-30:-24]))

# Returns the sensor value of a sample. Refer to step 2.
def get_values(str):
    return float(str.split(';')[2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', default= 3.6, help='Python compatibility (just for Alex, sorry!)', type = float)
    parser.add_argument('--csvpath', default= os.getcwd() + '/resources/trainingData/', help='CSV path for "local" option', type = str)
    parser.add_argument('--hdfspath', default= 'hdfs://localhost:9000/user/chusantonanzas/sensordata', help='HDFS path', type = str)

    args = parser.parse_args()

    version = args.version
    loadfrom = args.loadfrom
    csv_path = args.csvpath
    hdfs_path = args.hdfspath

    # configure env. Refer to step 1
    sc = config.config_env(version)
    sess = SparkSession(sc)

    # Refer to step 2.
    # Read csv's, and do averages for each flight date. Save in a DF with cols
    # ["aircraftid", "date"] ["average"]
    csvs = (sc.wholeTextFiles(csv_path+"*.csv")
              .flatMapValues(lambda t: t.split('\n'))
              .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
              .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
              .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
              .mapValues(lambda t: t[0]/t[1])).toDF()

    csvs.show(5, False)

    # load them into HDFS as AVRO. Refer to step 3.
    csvs.write.mode("overwrite").format("avro").save(hdfs_path)
    print(f'Sensor data saved in {hdfs_path}')
