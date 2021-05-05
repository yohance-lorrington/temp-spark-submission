# -*- coding: utf-8 -*-
import csv
import datetime
import json
import numpy as np
from pyspark.sql import SQLContext
import pyspark
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
# added line here
def calculateDate(x,dayOffset):
  start_date =  datetime.datetime.strptime(x[1][:10], "%Y-%m-%d")
  return (start_date + datetime.timedelta(days=dayOffset)).strftime("%Y-%m-%d")
def handleTuple(mappedTuple):
  if len(mappedTuple) >1:
    return (mappedTuple[0],json.loads(mappedTuple[1]))

def toCSVLine(data):
  return ','.join(str(d) for d in data)

if len(sys.argv) < 2:
    sys.exit(0)
sc = pyspark.SparkContext()

items = sc.broadcast([
    ('big_box_grocers',['452210','452311']),
    ('convenience_stores',['445120']),
    ('drinking_places',['722410']),
    ('full_service_restaurants',['722511']),
    ('limited_service_restaurants',['722513']),
    ('pharmacies_and_drug_stores',['446110','446191']),
    ('snack_and_bakeries',['311811','722515']),
    ('specialty_food_stores',['445210','445220','445230','445291','445292','445299']),
    ('supermarkets_except_convenience_stores',['445110'])
])

for item in items.value:
  unique_ids = sc.textFile('hdfs:///data/share/bdm/core-places-nyc.csv') \
    .map(lambda x: x.split(',')) \
    .map(lambda x: (x[1],x[9])) \
    .filter(lambda x: (x[1] in item[1])) \
    .map(lambda x: (x[0],1)) \
    .distinct() 
  
  loaddata = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*',use_unicode=False) \
    .map(lambda x: next(csv.reader([x]))) \
    .map(lambda x: (x[1],(x[12],x[16])))
  
  combined = unique_ids.join(loaddata) \
    .map(lambda x: x[1][1]) \
    .map(handleTuple) \
    .flatMap(lambda x:  [(calculateDate(x,index),visit)  for  index,visit in enumerate(x[1])] ) \
    .groupByKey() \
    .map(lambda x: (x[0],list(x[1]))) \
    .map(lambda x: (x[0][0:4],x[0],x[1])) \
    .map(lambda x: (x[0],x[1],min(x[2]),max(x[2]),np.median(x[2]))) \
    .map(toCSVLine)
    
  header = sc.parallelize(['year','date','low','high','median'])
  header.union(combined).saveAsTextFile('{}/{}'.format(sys.argv[1],item[0]))  
