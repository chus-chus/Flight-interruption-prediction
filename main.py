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

    # read from 'DW' aircraft utilization metrics and create response variable
    matrix = DMpipe.read_aircraft_util(sc)

    i = 0
    nyes = 0
    no = 0
    for x in matrix.collect():
        print(x)
        if x[1][4] == 'yes': nyes = nyes + 1
        elif x[1][4] == 'no': no = no + 1
    print(matrix.count())
    print("yes:", nyes,"no:", no)

    # Train and validate model
    model = DApipe.trainModel(matrix, sc)
