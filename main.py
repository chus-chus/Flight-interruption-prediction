import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

import data_management as DMpipe

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.7"
PYSPARK_DRIVER_PYTHON = "python3.7"

if(__name__== "__main__"):
    os.environ["HADOOP_HOME"] = HADOOP_HOME
    sys.path.append(HADOOP_HOME + "\\bin")
    os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
    os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_DRIVER_PYTHON

    conf = SparkConf()  # create the configuration
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    # read from 'DW' aircraft utilization metrics and create response variable
    ACutil = DMpipe.read_aircraft_util(sc)

    i = 0
    nyes = 0
    no = 0
    for x in ACutil.collect():
        if i < 100:
            print(x)
            i = i+1
        if x[1][4] == 'yes': nyes = nyes + 1
        elif x[1][4] == 'no': no = no + 1
    print("yes:", nyes,"no:", no)
    # 19560 2085

    # prueba 3 github
