import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

import data_management as DMpipe
import data_analysis as DApipe

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.6"
PYSPARK_DRIVER_PYTHON = "python3.6"

if(__name__== "__main__"):

    # Python compatibility (just for Alex, sorry!)
    if (len(sys.argv) == 2 and sys.argv[1] == 'a'):
        PYSPARK_PYTHON = PYSPARK_DRIVER_PYTHON = "python3.7"

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

    def show(rdd, len=100000, i=4):
        yes, no, k = 0, 0, 0
        for row in rdd.collect():
            if k < len: print(row); k += 1
            if row[i] == 1: yes += 1
            elif row[i] == 0: no += 1
        print(f'RDD with {rdd.count()} rows. [yes: {yes}, no: {no}]')

    ####
    # Pipelines
    ####

    # Read from 'DW' aircraft utilization metrics and create response variable.
    matrix = DMpipe.read_aircraft_util(sc)
    show(matrix, 5)

    # Train and validate model
    model = DApipe.trainModel(1, sc)
