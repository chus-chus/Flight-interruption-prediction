# Get csvs from hdfs, remember to deploy avro dependencies "org.apache.spark:spark-avro_2.11:2.4.3"
# for this option previous execution of "load_into_hdfs.py" is required.

import pyspark
import sys
import config
import os

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession


# converts to datetime.date format
def date_format(str):
    return '-'.join(("20"+str[4:6],str[2:4],str[0:2]))

def right_key(str):
    # Returns a KEY in ('aircraftid','dateid') format.
    return (str[-10:-4], date_format(str[-30:-24]))

def get_values(str):
    # Returns the sensor value of a sample.
    return float(str.split(';')[2])

if __name__ == "__main__":

    csv_path = os.getcwd() + '/resources/trainingData/'
    hdfs_path = 'hdfs://localhost:9000/user/chusantonanzas/sensordata'

    # Python compatibility (just for Alex, sorry!)
    version = 'python3.7' if (len(sys.argv) == 2 and sys.argv[1] == 'a') else 'python3.6'

    # configure env.
    sc = config.config_env(version)
    sess = SparkSession(sc)

    # read csv's, and do averages for each flight date. Save in a DF with cols
    # ["aircraftid", "date"] ["average"]
    csvs = (sc.wholeTextFiles(csv_path+"*.csv")
              .flatMapValues(lambda t: t.split('\n'))
              .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
              .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
              .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
              .mapValues(lambda t: t[0]/t[1])).toDF()

    csvs.show(5, False)

    # load them into HDFS as AVRO
    csvs.write.mode("overwrite").format("avro").save(hdfs_path)
