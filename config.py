import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

def config_env(version):

    HADOOP_HOME = "./resources/hadoop_home"
    JDBC_JAR = "./resources/postgresql-42.2.8.jar"
    PYSPARK_PYTHON = PYSPARK_DRIVER_PYTHON = version

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

    return sc
