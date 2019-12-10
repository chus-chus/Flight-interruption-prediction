import os
import pandas as pd
import pyspark
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
import numpy as np
import datetime

csv_path = os.getcwd() + '/resources/trainingData/'

DWuser = "jesus.maria.antonanzas"
DWpass = "DB200598"

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

attributes = ['aircraftid','dateid','sensorAVG']

def att(row,attribute):
    # Returns cell value given a row and an attribute: (row,col).
    for i, att in enumerate(attributes):
        if (attributes[i] == attribute):
            return row.split(',')[i]
    return None

# Marks flights 7 days before an unsch. maint. event.
def mark_days_before(t, it):
    nextval = it + 1
    distance = (t[it][0] - t[nextval][0]).days
    while distance <= 6:
        if len(t[nextval]) == 5: t[nextval].append("yes")
        nextval = nextval + 1
        # cannot get past last element
        if nextval == len(t):
            break
        else:
            distance = (t[it][0] - t[nextval][0]).days
    return

# Given a list of lists [date (descending), unscheduledoutofservice], marks, for each
# day, if there is an unscheduled maintenance event in the next 7 days ('yes')
# or not ('no'). The marks substitute 'unscheduledoutofservice'.
def add_response(t):
    t = list(t)
    for it in range(len(t)):
        # If the day is not visited (nothing appended), either it's the last
        # recorded day (we have no info) or there isn't an unsch. maint. event
        # in the next seven days. We do this after the first part because
        # we are replacing 'unscheduledoutofservice'.
        if len(t[it]) == 5: t[it].append("no")

        # if in a day there's an unsch. maint. event, mark all 7 days before it
        # (if not marked already). No need to loop if at the end of the list
        if t[it][1] == 1 and it < (len(t)-1): mark_days_before(t, it)
    return t

def right_key(str):
    # Returns a KEY in ('aircraftid','dateid') format.
    return (str[-10:-4], date_format(str[-30:-24]))

def get_values(str):
    # Returns the sensor value of a sample.
    return float(str.split(';')[2])

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

    maintenance = (load_amos
                   .map(lambda t: ((t[0], t[1].date()), t[2])))

    # For each flight, mark if there has been unscheduled maintenance sometime
    # in the next seven days.
    ACutilization = (load_dw
                     # select aircraftid, timeid, unscheduledoutofservice, FH, FC, DM
                     .map(lambda t: (t[0], [t[1], round(t[3]), round(float(t[2]), 2), int(t[4]), int(t[5])]))
                     # sort by timeid
                     .sortBy(lambda t: t[1][0], ascending = False)
                     # # get all flights of each aircraft
                     .groupByKey()
                     # # .map(lambda t: (t[0], add_response(t[1]))) no es necessari
                     # # ungroup (com podem estalviar-nos el map seguent?)
                     .flatMapValues(lambda t: add_response(t))
                     # posar-ho bé: ((aircraftid, time), (FH, FC, DM, response))
                     # t[1][1] (unscheduledoutofservice) is here for debugging !!!!!!!
                     .map(lambda t: ((t[0], t[1][0].strftime('%Y-%m-%d')), \
                     (t[1][2], t[1][3], t[1][4], t[1][5]))))

    #####
    # Adding sensors data
    #####

    # Get the average sensor values rdd: e.g. (('XY-SFN', '2014-12-04'), 60.624)
    avg = (sc.wholeTextFiles(csv_path+"*.csv")
        .flatMapValues(lambda t: t.split('\n'))
        .filter(lambda t: 'date' not in t[1] and len(t[1]) != 0)
        .map(lambda t: (right_key(t[0]), (get_values(t[1]), 1)))
        .reduceByKey(lambda t1,t2: (t1[0]+t2[0], t1[1]+t2[1]))
        .mapValues(lambda t: t[0]/t[1]))

    # return join
    return avg
