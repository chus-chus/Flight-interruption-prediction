import os
import sys
import pyspark
from pyspark import SparkConf
from pyspark.sql import SparkSession

HADOOP_HOME = "./resources/hadoop_home"
JDBC_JAR = "./resources/postgresql-42.2.8.jar"
PYSPARK_PYTHON = "python3.6"
PYSPARK_DRIVER_PYTHON = "python3.6"


if __name__ == "__main__":

    # Python compatibility (just for Alex, sorry!)
    if (len(sys.argv) == 2 and sys.argv[1] == 'a'):
        PYSPARK_PYTHON = PYSPARK_DRIVER_PYTHON = "python3.7"

def config_env():

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
