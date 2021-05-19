from pyspark import SparkContext
import datetime
import csv
import functools
import json
import numpy as np
import sys
import math
from functools import reduce
CAT_CODES = {'445210', '445110', '722410', '452311', '722513', '445120', '446110', '445299', '722515', '311811', '722511', '445230', '446191', '445291', '445220', '452210', '445292'}
CAT_GROUP = {'452210': 0,
             '452311': 0,
             '445120': 1,
             '722410': 2,
             '722511': 3,
             '722513': 4,
             '446110': 5,
             '446191': 5,
             '722515': 6,
             '311811': 6,
             '445210': 7,
             '445299': 7,
             '445230': 7,
             '445291': 7,
             '445220': 7,
             '445292': 7,
             '445110': 8}

def filterPOIs(_, lines):
  next(lines)
  items = map(lambda x:x.split(','),lines)
  items = map(lambda x: (x[0],x[9]), items)
  items = filter(lambda x: x[1] in CAT_CODES,items)
  items = map(lambda x: (x[0], CAT_GROUP[x[1]]),items)

  return items
def calculateDateNUM(x,dayOffset):
  start_date =  datetime.datetime.strptime(x[1][:10], "%Y-%m-%d")
  x = datetime.datetime(2019, 1, 1)
  return (start_date  - x ).days + dayOffset
def getTimeGreaterThan2018(x):
  start_date = datetime.datetime.strptime(x[1][:10], "%Y-%m-%d")
  x = datetime.datetime(2018, 12, 31)
  return start_date >= x
def extractVisits(storeGroup, _, lines):
  next(lines)
  items = map(lambda x: next(csv.reader([x])),lines) 
  items = map(lambda x: (x[0],x[12],x[14],x[16]),items)
  items = filter(lambda x: x[0] in storeGroup,items)
  items = filter(getTimeGreaterThan2018,items)
#   items = map(lambda x: (x[0],x[1],x[2],json.loads(x[3])),items)
  items = map(lambda x:  [(calculateDateNUM(x,index),visit,x[0])  for  index,visit in enumerate(json.loads(x[3]))],items )
  items = reduce((lambda arr1, arr2: arr1 + arr2), items)
  items = filter(lambda x: x[0]>=0,items)
  items = map(lambda x: ((storeGroup[x[2]],x[0]),x[1]),items)
  return items


def handleMedian(groupcounts,tupleObject):
  len_list = len(tupleObject[1])
  numberofzeros = groupcounts - len_list
  formatted_array = np.pad(tupleObject[1],(numberofzeros,),mode='constant',constant_values=(0,))
  std_dev = np.std(tupleObject[1])
  medianV = np.median(formatted_array)
  minV = max(0,medianV - std_dev)
  maxV = max(0,medianV + std_dev)
 
  return (tupleObject[0],medianV,minV,maxV)
def toCSVLine(data):
  return ','.join(str(d) for d in data)
def makeTimeStamp(dateDifferenceInt):
  start_date = datetime.datetime(2019, 1, 1)
  end_day = start_date + datetime.timedelta(days=dateDifferenceInt)
  return "2020-"+end_day.strftime("%m-%d")
def return2019or2020(num):
  return '2019' if num <365 else '2020'
def main(sc):
  '''
  Transfer our code from the notebook here, however, remember to replace
  the file paths with the ones provided in the problem description.
  '''
  rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
  rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
  OUTPUT_PREFIX = sys.argv[1]


  rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs) \
          .cache()
  storeGroup = dict(rddD.collect())
  groupCount = rddD \
    .map(lambda x: (x[1],1)) \
    .reduceByKey(lambda x,y: x+y) \
    .sortByKey(ascending=True) \
    .map(lambda x: x[1]) \
    .collect()
  rddG = rddPattern \
    .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))
  rddI = rddG.groupByKey() \
        .map(lambda x: (x[0],sorted(list(x[1])))) \
        .map(lambda x: handleMedian(groupCount[x[0][0]],x)) \
        .map(lambda x: (x[0][0],return2019or2020(x[0][1]),makeTimeStamp(x[0][1]),x[1],x[2],int(x[3]))) \
        .map(lambda x : (x[0],toCSVLine(x[1:6])))
  rddJ = rddI.sortBy(lambda x: x[1][:15])
  header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
  rddJ = (header + rddJ).coalesce(10).cache()
  
  items = sc.broadcast([
    ('big_box_grocers',0),
    ('convenience_stores',1),
    ('drinking_places',2),
    ('full_service_restaurants',3),
    ('limited_service_restaurants',4),
    ('pharmacies_and_drug_stores',5),
    ('snack_and_bakeries',6),
    ('specialty_food_stores',7),
    ('supermarkets_except_convenience_stores',8)
  ])
  for item in items.value:
    rddJ.filter(lambda x: x[0]==item[1] or x[0]==-1).values() \
      .saveAsTextFile(f'{OUTPUT_PREFIX}/{item[0]}')


if __name__=='__main__':
  sc = SparkContext()
  main(sc)
