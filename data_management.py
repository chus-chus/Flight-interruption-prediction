import os
import pandas as pd
import pyspark
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
import numpy as np
from datetime import datetime, timedelta

csv_path = os.getcwd() + '/resources/trainingData/'

DWuser = "jesus.maria.antonanzas"
DWpass = "DB200598"

attributes = ['aircraftid','dateid','sensorAVG']

def att(row,attribute):
    # Returns cell value given a row and an attribute: (row,col).
    for i, att in enumerate(attributes):
        if (attributes[i] == attribute):
            return row.split(',')[i]
    return None

# converts to datetime.date format
def date_format(str):
    # datetime.datetime.strptime("20"+str[4:6]+"-"+str[2:4]+"-"+str[0:2], '%Y-%m-%d').date()
    # print("20"+str[4:6]+"-"+str[2:4]+"-"+str[0:2])
    return '-'.join(("20"+str[4:6],str[2:4],str[0:2]))

def sensor_avg(path):
    data = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(path + file, sep = ';')
        aircraftid = file[-10:-4]
        timeid = date_format(file)
        avg = sum(df['value']) / len(df)
        # data = data.append({0: aircraftid, 1: timeid,
        #                     2: avg}, ignore_index=True)
        data = data.append({'aircraftid': aircraftid,'dateid': timeid,
                            'sensorAVG': avg}, ignore_index=True)
    return data

def right_key(str):
    # Returns a KEY in ('aircraftid','dateid') format.
    return (str[-10:-4], date_format(str[-30:-24]))

def get_values(str):
    # Returns the sensor value of a sample.
    return float(str.split(';')[2])

def create_priordays(pair):
    date = pair[0][1]
    priordays = []
    for i in range(1,7): priordays.append((date - timedelta(i),'yes'))
    return tuple(priordays)

def response(value):
    return "no" if value is None else "yes"


def read_aircraft_util(sc):
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

    load_amos = amos.select("aircraftregistration", "starttime", "subsystem").rdd

    # we pick the right sensor from AMOS database and generate artifical dates,
    # which we use to add the response variable
    maintenance = (load_amos
                   .map(lambda t: ((t[0], t[1].date()), t[2]))
                   .filter(lambda t: t[1] == "3454")
                   # generate 6 dates before each unshceduled maintenance. INCLOIM EL DIA QUE MIREM??????????
                   .map(lambda t: (t[0][0], create_priordays(t)))
                   .flatMapValues(lambda t: t)
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


    # Get the average sensor values rdd: e.g. (('XY-SFN', '2014-12-04'), 60.624)
    averages = (sc.wholeTextFiles(csv_path+"*.csv")
                  .flatMapValues(lambda t: t.split('\n'))
                  .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
                  .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
                  .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
                  .mapValues(lambda t: t[0]/t[1]))

    # final data matrix, ((aircraft, date), (FH, FC, DM, avg(sensor), response))
    matrix = (ACuti_Mevents.join(averages)
                           .mapValues(lambda t: (t[0][0], t[0][1], t[0][2], t[1], t[0][3])))

    return matrix
