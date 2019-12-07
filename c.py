import os
import pandas as pd
from datetime import date

# csv_path = os.getcwd() + '/resources/trainingData/'
csv_path = os.getcwd() + '/resources/t/'

def date_format(str):
    return date.fromisoformat('-'.join(('20'+str[4:6],str[2:4],str[0:2])))

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

print(sensor_avg(csv_path))
