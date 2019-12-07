from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql import SparkSession
import numpy as np

DWuser = "jesus.maria.antonanzas"
DWpass = "DB200598"

# Marks flights 7 days before an unsch. maint. event.
def mark_days_before(t, it):
    nextval = it + 1
    distance = (t[it][0] - t[nextval][0]).days
    while distance <= 7:
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
    it = 0
    for it in range(0, len(t)):
        # if in a day there's an unsch. maint. event, mark all 7 days before it
        # (if not marked already). No need to loop if at the end of the list
        #print("comprobando", t[it])
        if t[it][1] == 1 and it < (len(t)-1): mark_days_before(t, it)

        # If the day is not visited (nothing appended), either it's the last
        # recorded day (we have no info) or there isn't an unsch. maint. event
        # in the next seven days. We do this after the first part because
        # we are replacing 'unscheduledoutofservice'.
        if len(t[it]) == 5: t[it].append("no")
        it = it + 1
    return t

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

    load = dw.select("aircraftid", "timeid", "flighthours",
                     "unscheduledoutofservice", "flightcycles", "delayedminutes").rdd

    # For each flight, mark if there has been unscheduled maintenance sometime
    # in the next seven days.
    ACutilization = (load
                     # select aircraftid, timeid, unscheduledoutofservice, FH, FC, DM
                     .map(lambda t: (t[0], [t[1], round(t[3]), round(float(t[2]), 2), int(t[4]), int(t[5])]))
                     # sort by timeid
                     .sortBy(lambda t: t[1][0], ascending = False)
                     # get all flights of each aircraft
                     .groupByKey()
                     # .map(lambda t: (t[0], add_response(t[1]))) no es necessari
                     # ungroup (com podem estalviar-nos el map seguent?)
                     .flatMapValues(lambda t: add_response(t))
                     # posar-ho bé: ((aircraftid, time), (FH, FC, DM, response))
                     # t[1][1] (unscheduledoutofservice) if here for debugging !!!!!!!
                     .map(lambda t: ((t[0], t[1][0]), (t[1][2], t[1][3], t[1][4], t[1][1], t[1][5]))))


    return ACutilization
