"""
Environment configuration script
@author: Jesus Antonanzas Acero, Alex Carrillo Alza
@version: "1.0"
@email: "jesus.maria.antonanzas@est.fib.upc.edu, alex.carrillo.alza@est.fib.upc.edu"
@info: BDA, GCED, Big Data Analytics project
@date: 16/12/2019

Description
-----------
Configuration of Spark environment given the structure of the code skeleton provided.
This script is called in many parts of the project as a space saver, so that code is
not repeated.

Steps
-----------
1. Given a python version (string, i.e. "python3.6"), returns an object of type "Spark Context".
"""
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

    # start the configuration

    conf = SparkConf()
    conf.set("spark.jars", JDBC_JAR)

    spark = SparkSession.builder \
        .config(conf=conf) \
        .master("local") \
        .appName("Training") \
        .getOrCreate()

    sc = pyspark.SparkContext.getOrCreate()

    return sc
