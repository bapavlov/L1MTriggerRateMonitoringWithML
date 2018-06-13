#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-

import os
import sys
import time
import re

import cx_Oracle
import string

if len(sys.argv)<2 :
   print "Usage:",sys.argv[0]," run number, limit"
   exit(0)

limit = -1
if(len(sys.argv)>2):
    print sys.argv[0], ", Run:", sys.argv[1], ", Limit:",  sys.argv[2]
    limit = int(sys.argv[2])

connstr='cms_twinmux_mon/twinmux_MON_2k18@cms_omds_adg'
conn = cx_Oracle.connect(connstr)
curs = conn.cursor()
curs.arraysize=5000

query="SELECT RUNNUMBER, TIMESTAMP, ID, ALGO_RPC_RATE_1, ALGO_RPC_RATE_2, ALGO_RPC_RATE_3, ALGO_RPC_RATE_4,\
ALGO_ST_RATE_1, ALGO_ST_RATE_2, ALGO_ST_RATE_3, ALGO_ST_RATE_4, ALGO_ST_RATE_5 FROM PROCESSOR_TWINMUXPROCESSOR,\
TWINMUX_CELL WHERE RUNNUMBER='"+sys.argv[1]+"' and UNIQUEID=TWINMUX_CELL_UNIQUEID order by TIMESTAMP asc"
#print query

curs.execute(query)
debug=1
key=sys.argv[0]
ii=0
myList = []
#print "{ "
#print "RunNumber","Timestamp","Board","RPC_RATE_1","RPC_RATE_2","RPC_RATE_3","RPC_RATE_4",\
#"ST_RATE_1","ST_RATE_2","ST_RATE_3","ST_RATE_4","ST_RATE_5"
j = 0
for rows in curs:
     path=rows[0]
     description=rows[1]
     contacts=rows[2]
     #print rows[0],rows[1],rows[2],rows[3],rows[4],rows[5],rows[6],rows[7],rows[8],rows[9],rows[10],rows[11]
     myList.append([rows[0],rows[1],rows[2],rows[3],rows[4],rows[5],rows[6],rows[7],rows[8],rows[9],rows[10],rows[11]])
     j = j + 1
     if (j == limit):
         print "Reached limit:", j
         break

import csv
name = 'dt_rates_'+sys.argv[1]+'.csv'
print name
with open(name, 'wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for val in myList:
        wr.writerow(val)

