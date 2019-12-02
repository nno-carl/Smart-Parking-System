

import pymysql

pymysql.install_as_MySQLdb()

import MySQLdb
import pymongo
import datetime
import json
from pymongo import MongoClient

def judge(pp):
    client = MongoClient("localhost",27017,maxPoolSize=50)
    db = client.plateinfo
    collection = db.allinfo
    check = collection.find({"plate":pp})
    if check.count()>0:
        return True
    else:
        return False
    
def insert(pp,date,enterTime,leaveTime):
    client = MongoClient("localhost",27017,maxPoolSize=50)
    db = client.plateinfo
    collection = db.allinfo
    check = collection.find({"plate":pp})
    
    if check.count() > 0:
        collection.update({"plate": pp},
                             {"$set":{
                                     "leaveTime":leaveTime,
                                 }}
                             
    ) 
    

        enterTime = collection.find({"plate":pp},{"enterTime":1,"_id":0})
        leaveTime = collection.find({"plate":pp},{"leaveTime":1,"_id":0})
        for x in enterTime:
            print(x)
        for y in leaveTime:
            print(y)
        enterTime_list = list(x.values())
        enterTime = enterTime_list[0]
        leaveTime_list = list(y.values())
        leaveTime = leaveTime_list[0]
        print(enterTime)
        print(leaveTime)
        entertime_datetime = datetime.datetime.strptime(enterTime,'%Y-%m-%d-%H:%M:%S')
        leavetime_datetime = datetime.datetime.strptime(leaveTime,'%Y-%m-%d-%H:%M:%S')
        difference = (leavetime_datetime - entertime_datetime)
        print("duration:")
        print(difference)

            
        
    else:
        difference = 0
        collection.insert({"plate": pp,
                           "date":date,
                           "enterTime": enterTime,
                           "leaveTime":leaveTime,
                           })
        print(enterTime)
#    print("enter time:" + enterTime)
    return enterTime,leaveTime,difference
    
    
    