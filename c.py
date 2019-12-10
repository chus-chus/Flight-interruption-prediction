import os
import pandas as pd
from datetime import date

# csv_path = os.getcwd() + '/resources/trainingData/'
csv_path = os.getcwd() + '/resources/t/'

def date_format(str):
    return '-'.join(('20'+str[4:6],str[2:4],str[0:2]))

def sensor_avg(path):
    data = pd.DataFrame()
    for file in os.listdir(path):
        df = pd.read_csv(path + file, sep = ';')
        aircraftid = file[-10:-4]
        timeid = date_format(file)
        # print(type(timeid))
        avg = sum(df['value']) / len(df)
        # data = data.append({0: aircraftid, 1: timeid,
        #                     2: avg}, ignore_index=True)
        data = data.append({'aircraftid': aircraftid,'dateid': timeid,
                            'sensorAVG': avg}, ignore_index=True)
        print(type(data['aircraftid']))
    return data

def right_key(str):
    # Returns a KEY in ('aircraftid','dateid') format.
    return (str[-10:-4], date_format(str[-30:-24]))

def get_values(str):
    # Returns the sensor value of a sample.
    return str.split(';')[2]

print(right_key('file:/Users/Alex/projecteBDA/bda-project/resources/trainingData/170910-OMO-FSZ-6583-XY-SFN.csv'))
tup = (('XY-SFN', '2015-05-28'), '2015-05-28 10:27:32.136;eventData;65.86854369650402')
print(get_values(tup[1]))
str = ''
print(len(str))
# print(sensor_avg(csv_path))
