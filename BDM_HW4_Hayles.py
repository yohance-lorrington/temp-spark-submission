import csv
import datetime
import json
import numpy as np
from pyspark.sql import SQLContext
import sys
import pyspark
#
def calculateDate(x,dayOffset):
  start_date =  datetime.datetime.strptime(x[1][:10], "%Y-%m-%d")
  return (start_date + datetime.timedelta(days=dayOffset)).strftime("%Y-%m-%d")

# if len(sys.argv) < 2:
#   return 
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
  unique_ids = set(sc.textFile('hdfs:///data/share/bdm/core-places-nyc.csv') \
    .map(lambda x: x.split(',')) \
    .map(lambda x: (x[1],x[9])) \
    .filter(lambda x: (x[1] in item[1])) \
    .map(lambda x: x[0]) \
    .collect())
  
  patterns = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*') \
    .map(lambda x: next(csv.reader([x]))) \
    .filter(lambda x: (x[1] in unique_ids)) \
    .map(lambda x: (x[1],x[12],json.loads(x[16]))) \
    .flatMap(lambda x:  [(calculateDate(x,index),[visit])  for  index,visit in enumerate(x[2])] ) \
    .reduceByKey(lambda a,b: a+b) \
    .map(lambda x: (x[0][:4],x[0],int(np.median(x[1])),min(x[1]),max(x[1]))) \
    .sortBy(lambda x:x[1])
  if  not patterns.isEmpty():
    sqlContext = SQLContext(sc)
    df = sqlContext.createDataFrame(patterns,['year','date','median','low','high'])
    df.write.format("csv").save('{}/{}/{}.csv'.format(sys.argv[1],item[0],item[0]))