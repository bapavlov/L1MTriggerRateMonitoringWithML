#!/usr/bin/env python
# coding: utf-8

# ## TOC:
# * [Reading CVS files](#first-bullet)
# * [Luminosity section](#second-bullet)
# * [Trigger rate section](#third-bullet)
# * [Model inference section](#fourth-bullet)

# In[1]:


import math
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import ndimage, misc
import datetime
from datetime import timedelta

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Conv1D, MaxPooling1D, AveragePooling1D, UpSampling1D, InputLayer

from scipy import ndimage, misc

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split,StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import LocalOutlierFactor


# In[2]:


# Change presentation settings
get_ipython().run_line_magic('matplotlib', 'inline')

matplotlib.rcParams["figure.figsize"] = (15.0, 8.0)
matplotlib.rcParams["xtick.labelsize"] = 16
matplotlib.rcParams["ytick.labelsize"] = 16
matplotlib.rcParams["axes.spines.left"] = True
matplotlib.rcParams["axes.spines.bottom"] = True
matplotlib.rcParams["axes.spines.right"] = True
matplotlib.rcParams["axes.spines.top"] = True
matplotlib.rcParams["axes.titlesize"] = 16
matplotlib.rcParams["figure.titlesize"] = 16
matplotlib.rcParams["axes.labelsize"] = 14
matplotlib.rcParams["legend.fontsize"] = 14


# In[3]:


runs = [302634, 302635, 306125]
lumi_directory = data_directory = "./lumi"
rates_directory = "./rates"


# # Reading cvs files <a class="anchor" id="first-bullet"></a>

# Reading instantaneous luminosities from the cvs file produced with brilcalc and saving into a pandas dataframe:

# In[4]:


df_rates = pd.DataFrame()
int_lumi2 = pd.DataFrame()
for run in runs:
    print("Loading %s" % run)
    path = "%s/lumi_%s.csv" % (lumi_directory, run)
    int_lumi2 = int_lumi2.append(pd.read_csv(path,
        names=["runfill", "ls", "time", "beamstatus", "energy", "delivered",\
               "recorded", "avgpu", "source"]), 
        ignore_index=True);    
    path = "%s/dt_rates_%s.csv" % (rates_directory, run)
    df_rates = df_rates.append(pd.read_csv(path, 
        names=["run", "time", "board", "RPC1", "RPC2", "RPC3", "RPC4",\
               "DT1", "DT2", "DT3", "DT4", "DT5"]), 
        ignore_index=True);
print("Done.")


# # Luminosity section <a class="anchor" id="second-bullet"></a>

# Dropping useless rows inherited from the lumi CVS file:

# In[5]:


int_lumi2["source"] = int_lumi2["source"].astype('str')
int_lumi2 = int_lumi2[int_lumi2["source"] != "nan"]
int_lumi2 = int_lumi2[int_lumi2["source"] != "source"]


# Splitting run:fill field and the start and end lumi sections:

# In[6]:


int_lumi2['run'], int_lumi2['fill'] = int_lumi2['runfill'].str.split(':', 1).str
int_lumi2['ls_start'], int_lumi2['ls_end'] = int_lumi2['ls'].str.split(':', 1).str


# Converting run to integer and luminosities to float:

# In[7]:


int_lumi2["run"] = int_lumi2["run"].astype('int')
int_lumi2["ls_start"] = int_lumi2["ls_start"].astype('int')
int_lumi2["ls_end"] = int_lumi2["ls_end"].astype('int')
int_lumi2["delivered"] = int_lumi2["delivered"].astype('float64')
int_lumi2["recorded"] = int_lumi2["recorded"].astype('float64') 


# Converting time stamp to datetime:

# In[8]:


def transform_time(data):
    from datetime import datetime
    time_str = data.time
    #print time_str
    datetime_object = datetime.strptime(time_str, "%m/%d/%y %H:%M:%S")
    #print datetime_object
    return datetime_object
int_lumi2["time"] = int_lumi2.apply(transform_time, axis=1);


# Creating end time column from the start time:

# In[9]:


int_lumi2["time_end"] = int_lumi2["time"]


# Finding the runs and their start and end times:

# In[10]:


boundaries = pd.DataFrame(columns=["run", "start", "end", "ls_start", "ls_end", "nLS"])
for i in runs:
    start = int_lumi2[int_lumi2["run"] == i]["time"]
    end = int_lumi2[int_lumi2["run"] == i]["time_end"]
    start_ls = int_lumi2[int_lumi2["run"] == i]["ls_start"]
    end_ls = int_lumi2[int_lumi2["run"] == i]["ls_end"]
    start =  start.reset_index(drop=True)
    end =  end.reset_index(drop=True)
    start_ls =  start_ls.reset_index(drop=True)
    end_ls =  end_ls.reset_index(drop=True)
    nLS = int(start_ls.iloc[-1]) - int(start_ls.iloc[0]) + 1
    print i, start.iloc[0], start.iloc[-1], start_ls.iloc[0], start_ls.iloc[-1], nLS
    boundaries = boundaries.append({"run": i, "start": start.iloc[0], "end": start.iloc[-1], 
                                   "ls_start": start_ls.iloc[0], "ls_end": start_ls.iloc[-1], "nLS": nLS}, 
                                   ignore_index = True)


# Reindexing the dataframe after removing some lines:

# In[11]:


int_lumi2.index = pd.RangeIndex(len(int_lumi2.index))


# In[12]:


print len(int_lumi2.index)


# Filling end time column:

# In[13]:


def addTimeOffSet(startdate):
    enddate = pd.to_datetime(startdate) + pd.DateOffset(seconds=23)
    return enddate

def shiftElement(df, boundaries):
    run0 = boundaries["run"].iloc[0]
    for index, rows in df.iterrows():
        run = rows["run"]
        nls = int(boundaries[boundaries["run"] == run]["nLS"])
        if(run > run0):
            nls = nls + index
        #print run, nls
        if((index < nls) & (index < len(int_lumi2.index)-1)):
            #print index, run, rows["time"], df["time"][index+1]
            df.loc[index, "time_end"] = df["time"][index+1]
        elif (index == len(int_lumi2.index)-1):
            #print index, run, rows["time"], addTimeOffSet(rows["time"])
            df.loc[index, "time_end"] = addTimeOffSet(rows["time"])
                    
shiftElement(int_lumi2, boundaries)


# In[14]:


#print int_lumi2["beamstatus"]
int_lumi2 = int_lumi2[int_lumi2["beamstatus"] == "STABLE BEAMS"]


# Plotting the instantaneous luminosities:

# In[15]:


def plot_inst_lumi(x_val, y_val, z_val, title):
    fig, ax = plt.subplots()
    plt.xlabel("Time")
    plt.ylabel(r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]")
    xfmt = mdates.DateFormatter('%y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.grid()
    fig.autofmt_xdate()
    plt.plot(x_val, y_val, 'ro-')
    plt.plot(x_val, z_val, 'bo-')
    plt.title(title)
    plt.legend(loc="best")
    plt.show();


# In[16]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 302634]["time"], 
               int_lumi2[int_lumi2["run"] == 302634]["delivered"], 
               int_lumi2[int_lumi2["run"] == 302634]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               (boundaries["run"].iloc[0], int_lumi2["fill"].iloc[0])))


# In[17]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 302635]["time"], 
               int_lumi2[int_lumi2["run"] == 302635]["delivered"], 
               int_lumi2[int_lumi2["run"] == 302635]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               (boundaries["run"].iloc[1], int_lumi2["fill"].iloc[0])))


# In[18]:


int_lumi2.to_csv("int_lumi_issues.csv", sep='\t')


# # Trigger rate section <a class="anchor" id="third-bullet"></a>

# Converting columns to proper data types:

# In[19]:


df_rates["time"] = pd.to_datetime(df_rates["time"])
df_rates["run"] = df_rates["run"].astype('int')
#print df_rates["time"]


# Splitting, converting and adding new columns:

# In[20]:


df_rates['wheel'], df_rates['sector'] = df_rates['board'].str.split('_', 1).str
df_rates["wheel"] = df_rates["wheel"].astype(str)
df_rates["sector"] = df_rates["sector"].astype(str)


# In[21]:


df_rates["wheel"].replace(regex=True,inplace=True,to_replace=r'YB',value=r'')
df_rates["sector"].replace(regex=True,inplace=True,to_replace=r'S',value=r'')
df_rates["wheel"] = df_rates["wheel"].astype('int')
df_rates["sector"] = df_rates["sector"].astype('int')
df_rates["ls"] = -1
df_rates["lumi"] = -1.0
df_rates["score"] = -1
df_rates.to_csv("df_rates.csv", sep='\t')


# Plotting the rate coming from one of the stations:

# In[22]:


def plot_rate_vs_time(df, x_val, y_val, z_val, title):
    df_temp = df.copy()
    crit = df_temp["board"] == z_val
    df_temp = df_temp[crit]
    fig, ax = plt.subplots()
    plt.xlabel("Time")
    plt.ylabel("Rate [Hz]")
    ax.xaxis_date()
    xfmt = mdates.DateFormatter('%d-%m-%y %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.grid()
    fig.autofmt_xdate()
    plt.plot(df_temp[x_val], df_temp[y_val], 'ro-')
    plt.title(title)
    plt.legend(loc="best")
    plt.show();

plot_rate_vs_time(df_rates[df_rates.run == 302634], "time", "DT1",                  "YB+1_S4", "Rates for Runs / Fill / Board: %s / %s / %s" %                   ("302634", int_lumi2["fill"].iloc[2], "YB+1_S4"))


# In[23]:


fig, ax = plt.subplots()
#ax.set_yscale('log')
plt.xlabel("Time")
plt.ylabel("Rate [Hz] / Inst. Lumi. [10$^{30}$ Hz/cm$^2$]")
plt.xlim(int_lumi2[int_lumi2["run"] == 302634]["time"].iloc[0],         int_lumi2[int_lumi2["run"] == 302634]["time"].iloc[30])
ax.xaxis_date()
xfmt = mdates.DateFormatter('%d-%m-%y %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
ax.grid()
fig.autofmt_xdate()
plt.plot(int_lumi2[int_lumi2["run"] == 302634]["time"],         int_lumi2[int_lumi2["run"] == 302634]["delivered"], 'ro-')
rule = (df_rates.run == 302634) & (df_rates.board == "YB+1_S3")
plt.plot(df_rates[rule]["time"], df_rates[rule]["DT1"], 'bo-')
plt.title("")
plt.legend(loc='best')
plt.show()


# Associating a LS and an instantaneous luminosity to each rate:

# In[24]:


#Just a backup copy
df_rates_backup = df_rates.copy()


# In[25]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# Removing the measurements taken before and after the start and end time reported by the brilcalc output. All the 60 boards are measured at the same time. In order to speed-up the association, we take just one board, the first one. This reduces the dataframe and the time needed to go though it by a factor of 60.

# In[26]:


time0 = boundaries["start"].iloc[0]
timeF = boundaries["end"].iloc[-1]
print time0, timeF
#print df_rates[(df_rates.time >= time0) & (df_rates.time <= timeF)]
df_rates = df_rates[(df_rates.time >= time0) & (df_rates.time <= timeF)]
rule = df_rates.duplicated(subset=["time"])
count = (rule == False).sum()
print "Duplicates:", rule.sum()
df_rates_noduplicates = df_rates[rule == False]
#print df_rates_noduplicates


# In[27]:


print len(df_rates_noduplicates)


# Assigning the LS and the inst. lumi. to the measurements for the selected board:

# In[28]:


def assignLS(df1, df2, boundaries):
    temp = df1.copy()
    j = 1
    for index1, row1 in df1.iterrows():
        run1 = row1["run"]
        time1 = row1["time"]
        #print index1, run1, time1
        ti = time1 - 2*pd.DateOffset(seconds=23)
        tf = time1 + 2*pd.DateOffset(seconds=23)
        indexes2 = df2[(df2.run == run1) & (df2.time > ti) & (df2.time < tf)].index
        #print indexes2
        for i in indexes2:
            if((time1 >= df2["time"].loc[i]) & (time1 < df2["time_end"].loc[i])):
                #print time1, df2["time"].loc[i], df2["time_end"].loc[i]
                if(j%1000 == 0): 
                    print j
                j = j + 1
                ls = df2["ls_start"].loc[i]
                lumi = df2["delivered"].loc[i]
                #print index1, run1, time1, ls, lumi
                temp.loc[index1, "ls"] = ls
                temp.loc[index1, "lumi"] = lumi
                break
    return temp

temp = assignLS(df_rates_noduplicates, int_lumi2, boundaries)
df_rates_noduplicates = temp


# Removing the few cases not assigned and that are still at -1:

# In[29]:


df_rates_noduplicates = df_rates_noduplicates[df_rates_noduplicates["ls"] > 0]
print len(df_rates_noduplicates)


# Save in a csv file:

# In[30]:


df_rates.to_csv("df_rates_issues.csv", sep='\t')
df_rates_noduplicates.to_csv("df_rates_nodup_issues.csv", sep='\t')


# In[31]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# Assign the LS and the inst. lumi. to all the 60 boards for each time:

# In[32]:


def assignLS_ext(df1, df2):
    temp = df1.copy()
    indexes = []
    for index in df2.index:
        if index%10000 == 0:
            print index
        time = df2["time"].loc[index]
        ls = df2["ls"].loc[index]
        lumi = df2["lumi"].loc[index]
        des = (temp["time"] == time)
        indexes = temp[des].index
        #print time, ls, indexes
        temp.loc[des, "ls"] = ls
        temp.loc[des, "lumi"] = lumi
    return temp
    
temp = assignLS_ext(df_rates, df_rates_noduplicates)


# Save another backup copy:

# In[33]:


df_rates = temp.copy()
#print df_rates[df_rates.ls <= 0]


# Removing measurements without LS assignment:

# In[34]:


df_rates_backup = df_rates.copy()
df_rates = df_rates[df_rates.ls > 0]
#print df_rates["ls"]


# In[35]:


fig, ax = plt.subplots()
#ax.set_yscale('log')
plt.xlabel("LS")
plt.ylabel("Rate [Hz] / Inst. Lumi. [10$^{30}$ Hz/cm$^2$]")
plt.xlim(30, 100)
ax.grid()
plt.plot(df_rates[df_rates["run"] == 302634]["ls"],         df_rates[df_rates["run"] == 302634]["lumi"], 'ro-')
rule = (df_rates.run == 302634) & (df_rates.board == "YB+1_S3")
plt.plot(df_rates[rule]["ls"], df_rates[rule]["DT1"], 'bo-')
plt.title("")
plt.legend(loc='best')
plt.show()


# In[36]:


#print df_rates[df_rates.ls <= 0]
df_rates.to_csv("df_rates_issues.csv", sep='\t')


# Averaging the rates associated to the same LS:

# In[37]:


df_boards = df_rates.copy()
df_boards = df_boards.groupby(['board']).size().reset_index(name='counts')
print len(df_boards)
#print df_boards


# Too slow to use all the measurements. Averaging over 10 LS:

# In[38]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# In[39]:


bunch = 10
def assignGroup(data, div = bunch):
    res = int(data/div)
    #print data, res
    return res

df_rates["group"] = df_rates["ls"]
df_rates["group"] = df_rates["group"].apply(assignGroup)


# In[40]:


temp1 = df_rates.groupby(['run', 'group', 'board', 'wheel', 'sector'])[["ls", "lumi", "RPC1", "RPC2", "RPC3", "RPC4", "DT1", "DT2", "DT3", "DT4", "DT5"]].mean().reset_index()

temp2 = df_rates.groupby(['run', 'group', 'board', 'wheel', 'sector'])[["lumi", "RPC1", "RPC2", "RPC3", "RPC4", "DT1", "DT2", "DT3", "DT4", "DT5"]].std().reset_index()

temp3 = df_rates.groupby(['run', 'group', 'board', 'wheel', 'sector'])[["lumi", "RPC1", "RPC2", "RPC3", "RPC4", "DT1", "DT2", "DT3", "DT4", "DT5"]].size().reset_index(name='counts')

temp2 = temp2.rename(index=str, columns={"lumi": "errLumi", "RPC1": "errRPC1", "RPC2": "errRPC2",                                         "RPC3": "errRPC3", "RPC4": "errRPC4", "DT1": "errDT1",                                         "DT2": "errDT2", "DT3": "errDT3",                                         "DT4": "errDT4", "DT5": "errDT5"})

cols_to_use2 = temp2.columns.difference(temp1.columns)
cols_to_use3 = temp3.columns.difference(temp1.columns)

temp2 = temp2[cols_to_use2]
temp3 = temp3[cols_to_use3]

#print temp1.iloc[100]
#print temp2.iloc[100]
#print temp3.iloc[100]

temp1.reset_index(drop=True, inplace=True)
temp2.reset_index(drop=True, inplace=True)
temp3.reset_index(drop=True, inplace=True)

df_rates = pd.concat([temp1, temp2, temp3], axis = 1)


# Calculating the errors on the mean values calculated in the previous step:

# In[41]:


import math
def applySqrt(data):
    return math.sqrt(data)

df_rates["counts"] = df_rates["counts"].apply(applySqrt)

for i in list(df_rates):
    if "err" in i:
        #print i
        df_rates[i] = df_rates[i]/df_rates["counts"]


# Check for null or NaN values:

# In[42]:


print df_rates.isnull().values.any()
null_columns=df_rates.columns[df_rates.isnull().any()]
print(df_rates[df_rates.isnull().any(axis=1)][null_columns].head())
df_rates = df_rates.fillna(0)
print(df_rates[df_rates.isnull().any(axis=1)][null_columns].head())


# In[43]:


#Another backup
#df_rates_backup = df_rates.copy()
#df_rates.to_csv("df_rates_issues.csv", sep='\t')


# In[44]:


#Restore backup
#df_rates = df_rates_backup.copy()


# In[45]:


print len(df_rates)


# Uncomment to check just one case:

# In[46]:


#for index, row in df_rates.iterrows():
    #if row["board"] == "YB0_S1":
        #print "Index:", index,", Run:", row["run"],", Board: ",row["board"],",\
        #LS: ",row["ls"],", Rate: ",row["DT1"],", Error: ",row["errDT1"]


# Plotting the result:

# In[47]:


def plot_ratio_vs_ls(df, run, x_val, y_val, z_val, x_err, y_err, title_x, title_y, title, opt, log):
    df_temp = df.copy()
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    if log:
        ax.set_yscale('log')
    inter = []
    errors = []
    df_temp["ratio"] = df_temp[y_val]/df_temp[x_val]
    df_temp["errRatio"] = (1/df_temp[x_val])*    np.sqrt(df_temp[y_err]**2 + df_temp["errLumi"]**2*df_temp["ratio"]**2)
    for i in range(len(run)):
        rule = ((df_temp["board"] == z_val) & (df_temp["run"] == run[i]))
        val1 = df_temp[rule][x_val]
        val2 = df_temp[rule]["ratio"]
        inter.append(np.polyfit(val1, val2, 0)) #append anomalous fit,\
                                                #then normal fit, and ratio of the two
        errors.append(df_temp[rule]["errRatio"])
        #plt.plot(df_temp[rule][x_val], df_temp[rule]["ratio"], opt[i], label=str(run[i]))
        newstr = opt[i].replace("o", "")
        plt.errorbar(df_temp[rule][x_val], df_temp[rule]["ratio"], xerr=x_err,        yerr=df_temp[rule]["errRatio"], fmt=opt[i], ecolor=newstr, label=str(run[i]))
    a, b, c = inter[0], inter[1], 1-inter[0]/inter[1]
    plt.legend(loc="best")
    plt.title(title)
    plt.show()
    return float(a), float(b), float(c)


# In[48]:


ratio = pd.DataFrame(columns=["wheel", "sector", "station", "ratio", "ref", "anomal",                              "cs_norm", "cs_anomal"])
for i in [-2, -1, 0, +1, +2]:
    for j in range(1, 13):
        if ((j == 4) | (j == 10)):
            range_ = range(1, 6)
        else:
            range_ = range(1, 5)
        for k in range_:

            if (i > 0):
                board = "YB+" + str(i) + "_S" + str(j)
            else:
                board = "YB" + str(i) + "_S" + str(j)
            station = "DT" + str(k)
            
            error = "err"+station
            
            title = "Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+                            " / "+str("[302634, 306125]")+" / " + board + "_MB" + str(k)
                
            (a, b, c) = plot_ratio_vs_ls(df_rates, [302634, 306125], "lumi", station, board, 0,                            error, r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                            r"Cross-section [$\times10^{-30}$ cm$^{2}$]", title, ["ro", "bo"], False)

            ratio = ratio.append({"wheel": i, "sector": j, "station": k, "ratio": c, "ref": b,                                  "anomal": a, "cs_norm": 1/b, "cs_anomal": 1/a}, ignore_index = True)
            
            print "Anomalous CS:", a, ", Normal CS:", b, ", Ratio:", c
            


# In[49]:


(a1, b1, c1) = plot_ratio_vs_ls(df_rates, [306125, 306125], "lumi", "DT3", "YB+1_S4", 0,                            "errDT3", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                            r"Cross-section [$\times10^{-30}$ cm$^{2}$]", "W+1_S4_MB3",\
                                ["ro", "bo"], False)
(a2, b2, c2) = plot_ratio_vs_ls(df_rates, [306125, 306125], "lumi", "DT3", "YB-1_S3", 0,                            "errDT3", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                            r"Cross-section [$\times10^{-30}$ cm$^{2}$]", "W-1_S3_MB3",\
                                ["ro", "bo"], False)
print "Fit W+1_S4_MB3: ", a1, ", Fit W-1_S3_MB3:", a2, ", Ratio:", abs(a1-a2)/a2
value = abs(a1-a2)/a2


# In[50]:


def plot_ratio_vs_ls_2(df, run, x_val, y_val, z_val, x_err, y_err, title_x, title_y,                       title, opt, log, labels):
    df_temp = df.copy()
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    #plt.ylim(0, 5.)
    #plt.xlim(0, 4000)
    #ax.grid()
    if log:
        ax.set_yscale('log')
    df_temp["ratio"] = df_temp[y_val]/df_temp[x_val]
    df_temp["errRatio"] = (1/df_temp[x_val])*    np.sqrt(df_temp[y_err]**2 + df_temp["errLumi"]**2*df_temp["ratio"]**2)
    for i in range(0, len(z_val)):
        rule = ((df_temp["board"] == z_val[i]) & (df_temp["run"] == run))
        #plt.plot(df_temp[rule][x_val], df_temp[rule]["ratio"], opt[i], label=labels[i])
        newstr = opt[i].replace("o", "")
        plt.errorbar(df_temp[rule][x_val], df_temp[rule]["ratio"], xerr=x_err,        yerr=df_temp[rule]["errRatio"],        fmt=opt[i], ecolor=newstr, label=labels[i])
    plt.legend(loc="best")
    plt.title(title)
    plt.axhline(a1, color='r', linestyle='dashed', linewidth=1)
    plt.axhline(a2, color='b', linestyle='dashed', linewidth=1)
    plt.show()

title = "Fill/Run: "+str(int_lumi2["fill"].iloc[0])+"/"+str("306125")
plot_ratio_vs_ls_2(df_rates, 306125, "lumi", "DT3", ["YB+1_S4", "YB-1_S3"], 0,                            "errDT3", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                            r"Cross-section [$\times10^{-30}$ cm$^{2}$]",\
                   title, ["ro", "bo"], False,\
                   ["W+1_S4_MB3: anomalous chamber", "W-1_S3_MB3: normal symmetric chamber"])


# In[51]:


fig, ax = plt.subplots()
ax.set_yscale('log')
bins = np.linspace(0, 0.5, 100)
rule1 = (ratio.wheel == 1) & (ratio.sector == 4) & (ratio.station == 3)
rule2 = (ratio.wheel == -1) & (ratio.sector == 3) & (ratio.station == 3)
plt.hist(ratio["ratio"], bins=bins, alpha=0.5, label="Ratio: 302634 vs. 306125")
plt.hist(ratio[rule1]["ratio"], bins=bins, alpha=0.5, label="Ratio: W+1_S4_MB3")
plt.hist(ratio[rule2]["ratio"], bins=bins, alpha=0.5, label="Ratio: W-1_S3_MB3")
plt.title("")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Decrease in CS (w.r.t. 306125)')
plt.axvline(value, color='k', linestyle='dashed', linewidth=1)
plt.show()


# In[52]:


print "Total number of chambers:" ,len(ratio["ratio"].values)
count = np.where(ratio["ratio"].values > value)
print "Chambers over threshold: ", len(count[0])
print "Fraction of chambers over threshold:", float(len(count[0])) / len(ratio["ratio"].values)


# In[53]:


fig, ax = plt.subplots()
ax.set_yscale('log')
#ax.set_xscale('log')
bins = np.linspace(0, 2, 200)
rule1 = (ratio.wheel == 1) & (ratio.sector == 4) & (ratio.station == 3)
rule2 = (ratio.wheel == -1) & (ratio.sector == 3) & (ratio.station == 3)
plt.hist(ratio["ref"], bins=bins, alpha=0.5, label="Normal CS")
plt.hist(ratio["anomal"], bins=bins, alpha=0.5, label="Anomalous CS")
plt.hist(ratio[rule1]["ref"], bins=bins, alpha=0.5, label="Normal CS: W+1_S4_MB3")
plt.hist(ratio[rule2]["ref"], bins=bins, alpha=0.5, label="Normal CS: W-1_S3_MB3")
plt.hist(ratio[rule1]["anomal"], bins=bins, alpha=0.5, label="Anomalous CS: W+1_S4_MB3")
plt.hist(ratio[rule2]["anomal"], bins=bins, alpha=0.5, label="Anomalous CS: W-1_S3_MB3")
plt.title("")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel(r'Cross-section [$\times 10^{30}$ cm$^{2}$]')
plt.axvline(1.0, color='k', linestyle='dashed', linewidth=1)
plt.show()


# Create a new dataframe with the input features already organized in a numpy array:

# In[54]:


print len(df_rates)
algos = ['RPC1', 'RPC2', 'RPC3', 'RPC4', 'DT1', 'DT2', 'DT3', 'DT4', 'DT5']
df_rates_new = pd.DataFrame(columns=['run', 'group', 'board', 'wheel', 'sector', 'ls',                                     'lumi', 'errLumi', 'rate', 'err', 'system', 'station'])

for i in algos:
    list_a = ['run', 'group', 'board', 'wheel', 'sector', 'ls', 'lumi', 'errLumi', i, 'err'+i,]
    temp = df_rates.copy()
    temp = temp[list_a]
    temp["system"] = -1
    temp["station"] = -1
    j = i
    if (i.find("RPC") != -1):
        temp["system"] = 1
        num = i.replace("RPC", "")
        temp["station"] = int(num)
    else:
        temp["system"] = 2
        num = i.replace("DT", "")
        temp["station"] = int(num)
    temp = temp.rename(columns={j: 'rate', 'err'+j: 'err'})
    #print temp.columns
    df_rates_new = pd.concat([df_rates_new, temp], ignore_index=True)

print len(df_rates_new)


# In[55]:


def plot_rate_vs_ls(df, run, x_val, y_val, z_val, x_err, y_err, title_x, title_y, title, opt, log):
    df_temp = df.copy()
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    if log:
        ax.set_yscale('log')
    for i in range(len(run)):
        rule = ((df_temp["board"] == z_val) & (df_temp["run"] == run[i]))
        plt.errorbar(df_temp[rule][x_val], df_temp[rule][y_val], xerr=x_err,                     yerr=df_temp[rule][y_err], fmt=opt, ecolor='r')
    plt.legend(loc="best")
    plt.title(title)
    plt.show()


# In[56]:


title = "Rates for Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+                            " / 302634 / YB+1_S4_MB1"
plot_rate_vs_ls(df_rates_new[(df_rates_new["system"] == 2) & (df_rates_new["station"] == 1)], [302634],                "lumi", "rate", "YB+1_S4", 0, "err", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro", False)


# In[57]:


def plot_vs_ls(df, run, x_val, y_val, z_val, x_err, y_err, title_x, title_y, title, opt, log):
    df_temp = df.copy()
    fig, ax = plt.subplots()
    fig.autofmt_xdate()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    if log:
        ax.set_yscale('log')
    for i in range(len(run)):
        rule = ((df_temp["board"] == z_val) & (df_temp["run"] == run[i]))
        plt.errorbar(df_temp[rule][x_val], df_temp[rule][y_val], xerr=x_err, yerr=df_temp[rule][y_err],                     fmt=opt, ecolor='r')
        plt.errorbar(df_temp[rule][x_val], df_temp[rule]["lumi"], xerr=x_err, yerr=df_temp[rule]["errLumi"],                     fmt='bo', ecolor='b')
    plt.legend(loc="best")
    plt.title(title)
    plt.show()


# In[58]:


title = "Rates for Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+                            " / 302634 / YB+1_S4_MB4"
plot_vs_ls(df_rates_new[(df_rates_new["system"] == 2) & (df_rates_new["station"] == 4)], [302634],                "ls", "rate", "YB+1_S4", 0, "err", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro", False)


# In[59]:


title = "Rates for Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+                            " / 306125 / YB+1_S4_MB4"
plot_vs_ls(df_rates_new[(df_rates_new["system"] == 2) & (df_rates_new["station"] == 4)], [306125],                "ls", "rate", "YB+1_S4", 0, "err", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro", False)


# In[60]:


df_rates_new["CS"] = -1
df_rates_new["errCS"] = -1

df_rates_new["CS"] = df_rates_new["rate"]/df_rates_new["lumi"]
print "Number of NaN's in CS before:"
print len(df_rates_new["CS"][df_rates_new["CS"].isnull() == True])
print "Number of Inf's in CS before:"
print len(df_rates_new["CS"][np.isinf(df_rates_new["CS"])])

df_rates_new["CS"] = df_rates_new["CS"].replace([np.inf, -np.inf], np.nan)
df_rates_new["CS"] = df_rates_new["CS"].fillna(-1)

print "Number of NaN's in CS after:"
print len(df_rates_new["CS"][df_rates_new["CS"].isnull() == True])
print "Number of Inf's in CS after:"
print len(df_rates_new["CS"][np.isinf(df_rates_new["CS"])])

df_rates_new["errCS"] = (1/df_rates_new["lumi"])*np.sqrt(df_rates_new["err"]**2 + df_rates_new["CS"]**2 * df_rates_new["errLumi"]**2)

print "Number of NaN's in errCS before:"
print len(df_rates_new["errCS"][df_rates_new["errCS"].isnull() == True])
print "Number of Inf's in errCS before:"
print len(df_rates_new["errCS"][np.isinf(df_rates_new["errCS"])])

df_rates_new["errCS"] = df_rates_new["errCS"].replace([np.inf, -np.inf], np.nan)
df_rates_new["errCS"] = df_rates_new["errCS"].fillna(-1)

print "Number of NaN's in errCS after:"
print len(df_rates_new["errCS"][df_rates_new["errCS"].isnull() == True])
print "Number of Inf's in errCS after:"
print len(df_rates_new["errCS"][np.isinf(df_rates_new["errCS"])])


# In[61]:


array = df_rates_new.as_matrix(columns=['system', 'wheel', 'sector', 'station',                                        'lumi', 'errLumi', 'rate', 'err',                                        'CS', 'errCS'])


# In[62]:


print len(array)
print len(df_rates_new)


# In[63]:


df_rates_new["content"] = np.empty((len(df_rates_new), 0)).tolist()


# In[64]:


for index, rows in df_rates_new.iterrows():
    #print index, array[index]
    df_rates_new.at[index, "content"] = array[index]
df_rates_new["score"] = -1


# In[65]:


#print df_rates_new[(df_rates_new["station"] == 5) & ((df_rates_new["sector"] != 4) &\
 #                                                    (df_rates_new["sector"] != 10))]["content"]

rule = ((df_rates_new["station"] == 5) & ((df_rates_new["sector"] != 4) & (df_rates_new["sector"] != 10)))
df_rates_new = df_rates_new[rule == False]


# In[66]:


df_rates_new["ratio"] = 0
for i in [-2, -1, 0, +1, +2]:
    for j in range(1, 13):
        range_ = range(1, 5)
        if ((j == 4) | (j == 10)):
            range_ = range(1, 6)
        for k in range_:
            rule = (df_rates_new.wheel == i) & (df_rates_new.sector == j) & (df_rates_new.station == k)
            rule_r = (ratio.wheel == i) & (ratio.sector == j) & (ratio.station == k)
            idx = df_rates_new[rule].index
            value = ratio[rule_r]["ratio"].item()
            #print idx, value
            for w in idx:
                df_rates_new.loc[w, "ratio"] = value


# # Model inference section <a class="anchor" id="fourth-bullet"></a>

# ## Creating the train and test samples

# In[67]:


df_rates_new_2 = df_rates_new.copy()
print df_rates_new["content"].iloc[1]


# In[68]:


#To be run just one time!!!!!!
df_rates_new["content_in"] = df_rates_new["content"].copy()
def change_data(data):
    temp = data.copy()
    temp[2] = temp[2] - 6
    temp[3] = temp[3] - 3
    return temp

df_rates_new["content"] = df_rates_new["content"].apply(change_data)


# In[69]:


df_rates_new = df_rates_new[df_rates_new["system"] == 2]


# In[70]:


print df_rates_new["content"].iloc[1]


# In[71]:


normalies = df_rates_new.copy()
anomalies = df_rates_new.copy()
print len(normalies), len(anomalies)


# In[72]:


def deduceLS(data):
    if (bunch == 10):
        return data*10+5
anomalies["averageLS"] = anomalies["group"].apply(deduceLS)
#print anomalies.columns


# In[73]:


rule = (normalies["wheel"] == -1) & (normalies["sector"] == 3) & (normalies["station"] == 3) 
print "Normal chimney:"
print normalies[rule]["content"].iloc[100]

rule = (normalies["wheel"] == 1) & (normalies["sector"] == 4) & (normalies["station"] == 3)
print "Anomalous chimney:"
print normalies[rule]["content"].iloc[100]


# In[74]:


def assignScore(df, score):
    temp = df.copy()
    rule = (temp["wheel"] == 1) & (temp["sector"] == 4) & (temp["station"] == 3)
    indexes = temp[rule].index
    #print indexes
    for i in indexes:
        temp.loc[i, "score"] = score
    return temp

temp = assignScore(anomalies, 1)
anomalies = temp


# During runs 302634 (all LS') and 302635 (LS <= 404) one layer per station did not reach the correct HV, except for: MB1, MB4 in sectors 3, 4, 5 in wheels 2, -2, 0: 

# In[75]:


def assignScore2(df, score, run, ls_in, ls_end, ls_string):
    temp = df.copy()
    rule = (temp["run"] == run)
    if ((ls_in != -1) & (ls_end != -1)):
        rule = rule & (temp[ls_string] >= ls_in) & (temp[ls_string] <= ls_end)
    elif (ls_in != -1):
        rule = rule & (temp[ls_string] >= ls_in)
    elif (ls_end != -1):
        rule = rule & (temp[ls_string] <= ls_end)
    indexes = temp[rule].index
    for w in indexes:
        temp.loc[w, "score"] = score

    #Correct score for the chambers with normal HV
    for i in [-2, 0, 2]: #wheels that are ok
        for j in [3, 4, 5]: #sectors that are ok
            if (i == 0):
                range_ = [1]
            elif (abs(i) == 2):
                range_ = [1, 4, 5]
            else:
                range_ = [1, 4]
            for k in range_: #stations that are ok
                rule = (temp["run"] == run) & (temp["wheel"] == i)                         & (temp["sector"] == j) & (temp["station"] == k)
                indexes = temp[rule].index
                for w in indexes:
                    temp.loc[w, "score"] = -1
    return temp


# In[76]:


rule = ((anomalies["wheel"] == 2) & (anomalies["sector"] == 3) & (anomalies["station"] == 1))
print anomalies[rule]["score"].iloc[1]
temp = assignScore2(anomalies, 1, 302634, -1, -1, "group")
anomalies = temp
#temp = assignScore2(temp, 1, 302635, 1, 404, "group")
#anomalies = temp
print anomalies[rule]["score"].iloc[1]


# In[77]:


rule = ((anomalies["wheel"] == 1) & (anomalies["sector"] == 3) & (anomalies["station"] == 1))
print anomalies[rule]["score"].iloc[1]


# In[78]:


rule = ((anomalies["wheel"] == 2) & (anomalies["sector"] == 4) & (anomalies["station"] == 5))
print anomalies[rule]["score"].iloc[1]
rule = ((anomalies["wheel"] == -2) & (anomalies["sector"] == 4) & (anomalies["station"] == 5))
print anomalies[rule]["score"].iloc[1]


# In[79]:


def change(data, red = 1):
    #print data
    temp = data.copy()
    temp[4] = temp[4]*red #Increase lumi by "red"
    temp[5] = temp[5]*red
    temp[8] = temp[8]*red #Increase CS by "red"
    temp[9] = temp[9]*red
    #print temp
    return temp

print anomalies["content"].iloc[0]
anomalies["content_2"] = anomalies["content"].copy()
anomalies["content_2"] = anomalies["content"].apply(change)
print anomalies["content_2"].iloc[0]


# In[80]:


#Scale the data
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing.data import QuantileTransformer
def scale_data(data, scaler_type = 2):
    
    types = ["RobustScaler", "MaxAbsScaler", "QuantileTransformerUniform",             "QuantileTransformerNormal", "Normalizer", "MinMaxScaler"]
    #print "Scaling data using:", types[scaler_type-1]
    
    # Need to reshape since scaler works per column

    if (scaler_type == 0):
        return data
    if (scaler_type == 1):
        data = data.reshape(-1, 1)
        scaler = RobustScaler(quantile_range=(25, 75)).fit(data)
        return scaler.transform(data).reshape(1, -1)
    if (scaler_type == 2):
        data = data.reshape(-1, 1)
        scaler = MaxAbsScaler().fit(data)
        return scaler.transform(data).reshape(1, -1)
    if (scaler_type == 3):
        data = data.reshape(-1, 1)
        scaler = QuantileTransformer(output_distribution='uniform').fit(data)
        return scaler.transform(data).reshape(1, -1)
    if (scaler_type == 4):
        data = data.reshape(-1, 1)
        scaler = QuantileTransformer(output_distribution='normal').fit(data)
        return scaler.transform(data).reshape(1, -1)
    if (scaler_type == 5):
        data = data.reshape(1, -1)
        scaler = Normalizer(norm='l2').fit_transform(data)
        return scaler.reshape(1, -1)
    if (scaler_type == 6):
        data = data.reshape(-1, 1)
        scaler = MinMaxScaler().fit(data)
        return scaler.transform(data).reshape(1, -1)


# In[81]:


anomalies["content"] = anomalies["content"].apply(np.array)
anomalies["content_2"] = anomalies["content_2"].apply(np.array)
anomalies["content_scaled"] = anomalies["content"].apply(scale_data)
#anomalies["content_scaled"] = anomalies["content_2"].apply(scale_data)


# Testing on the high rate issue:

# In[82]:


#layers_test = anomalies.copy()
layers_test = (anomalies[(anomalies.run == 302634) | (anomalies.run == 306125)]).copy()
print "Tot:", len(layers_test)
print "Normalies:", len(layers_test[layers_test.score == -1])
print "Anomalies", len(layers_test[layers_test.score == 1])


# In[83]:


def score_to_array(score):
    if score == -1:
        return np.asarray([1, 0]) #Normaly
    return np.asarray([0, 1]) #Anomaly

def nn_generate_input():  
    return np.array(np.concatenate(layers_test.content_scaled.values)).reshape(-1, 10)

test_x = nn_generate_input()


# ## Simple test

# In[84]:


def simple_test(cross_sections, df, coeff):
    for i in df.index:
        wheel = df.wheel.loc[i]
        sector = df.sector.loc[i]
        station = df.station.loc[i]
        cs = df.CS.loc[i]
        #err_cs = df.errCS.loc[i]
        #print cs
        rule = ((cross_sections.wheel == wheel) & (cross_sections.sector == sector) &                (cross_sections.station == station))
        cs_norm = cross_sections[rule].ref.iloc[0]
        #err_cs_norm = cross_sections[rule].errCS.iloc[0]
        #low = cs_norm - coeff*err_cs_norm
        #hi = cs_norm + coeff*err_cs_norm
        low = cs_norm*(1 - coeff)
        hi = cs_norm*(1 + coeff)
        #print low, hi
        if ((cs > low) & (cs < hi)):
            res = 0
        else:
            res = +1
        df.loc[i, "st_score"] = res


# N.B.: Comparing wheel, sector and station is ok, because the scaling is applied only on the content

# In[85]:


layers_test["st_score"] = 0
layers_test.index = pd.RangeIndex(len(layers_test.index))
simple_test(ratio, layers_test, 0.05)


# In[86]:


# Distribution of scores:
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, +1, 3)
plt.hist(layers_test[layers_test["score"] < 0]["st_score"], bins=bins, alpha=0.5,         label="Normal chambers")
plt.hist(layers_test[layers_test["score"] > 0]["st_score"], bins=bins, alpha=0.5,         label="Anomalous chambers")
plt.title("Distribution of scores: Simple Test")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
plt.show()


# In[88]:


import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[89]:


th_km = 0.0
y_pred = 2*(layers_test["st_score"] > th_km)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix Simple Test, with normalization')


# In[90]:


def plotFpVsLs(run, wheel, sector, station, title, df, algo, threshold, log, bound):
    fig, ax = plt.subplots()
    if log:
        ax.set_yscale('log')
    bins = np.linspace(0, bound, int(bound/10)+10)
    (n, bins, patches) = plt.hist(df[(df["score"] == -1) & (df[algo] > threshold)
                         #& (df["wheel"] == 2) &\
                         #& (df["sector"] == 4) &\
                         #& (df["station"] == 2) &\
                         & (df["run"] == run)\
                        ]["averageLS"],
             bins=bins, alpha=0.5, label="False positives")
    plt.hist(df[(df["score"] == 1) & (df[algo] > threshold)
                         #& (df["wheel"] == 2) &\
                         #& (df["sector"] == 4) &\
                         #& (df["station"] == 2) &\
                         & (df["run"] == run)\
                        ]["averageLS"],
             bins=bins, alpha=0.5, label="True positives")
    plt.plot(df[(df["run"] == run)]["averageLS"],             df[(df["run"] == run)]["lumi"], "ro",             alpha=0.5, label="Inst. Lumi. $(x 10^{30} Hz/cm^2)$")
    plt.title(title+str(run))
    plt.legend(loc='best')
    plt.ylabel('Frequency')
    plt.xlabel('LS')
    plt.grid(True)
    #plt.plot([bound, bound], [0, 100], color='r', linestyle='--', linewidth=2)
    plt.show()
    return n


# In[91]:


layers_test["averageLS"] = layers_test["group"].apply(deduceLS)
threshold = 0.0
plotFpVsLs(302634, 0, 0, 0, "Distribution of false positives: Simple Test, ",           layers_test, "st_score",           threshold, True, boundaries[boundaries["run"] == 302634]["ls_end"])


# ## Inference

# Making an inference using the model and the test sample:

# In[92]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
K.set_session(sess)


# In[94]:


ann_model_1 = load_model("./model_keras/ann_1.h5")
ann_model_2 = load_model("./model_keras/ann_2.h5")
ann_model_3 = load_model("./model_keras/ann_3.h5")
ann_model_4 = load_model("./model_keras/ann_4.h5")


# In[95]:


layers_test["ann_score_1"] = ann_model_1.predict(np.array(test_x))[:, 1]
layers_test["ann_score_2"] = ann_model_2.predict(np.array(test_x))[:, 1]
layers_test["ann_score_3"] = ann_model_3.predict(np.array(test_x))[:, 1]
layers_test["ann_score_4"] = ann_model_4.predict(np.array(test_x))[:, 1]


# In[96]:


cae_model = load_model("./model_keras/cae.h5")
layers_test["cae_score"] = np.sum(abs(test_x - cae_model.predict(np.array(test_x))), axis=1)


# In[97]:


def benchmark(y_true, y_score, treshold):
    y_pred = 2*(y_score > treshold)-1
    y_true = 2*(y_true > treshold)-1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = round(float(tp)/(tp+fn), 4)
    specificity = round(float(tn)/(tn+fp), 4)

    print("Model accuracy: %s" % round(accuracy_score(y_true, y_pred), 4))
    print("Model sensitivity: %s" % sensitivity)
    print("Model specificity: %s" % specificity)

    return specificity, sensitivity
    
print "DNN:"
specificity_ann, sensitivity_ann = benchmark(layers_test["score"], layers_test["ann_score_4"], 0.5)
print "AE:"
specificity_cae, sensitivity_cae = benchmark(layers_test["score"], layers_test["cae_score"], 0.055)


# In[98]:


def get_roc_curve(test_df, models, working_point=None):
    """Generates ROC Curves for a given array"""
    fig, ax = plt.subplots()
    ax.grid()

    for legend_label, model_score in models:
        false_positive_rate, true_positive_rate, _ = roc_curve(test_df["score"],
                                                               test_df[model_score])
        #plt.xlim(0, 0.2)
        plt.plot(false_positive_rate, true_positive_rate, linewidth=2,
                 label=('%s, AUC: %s' % (legend_label,
                                         round(auc(false_positive_rate, true_positive_rate), 4))))
    if working_point:
        plt.plot(1-working_point[0],
                 working_point[1],
                 'o',
                 label="DNN working point")
    plt.title("ROC")
    plt.legend(loc='best')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    plt.show();


# In[99]:


get_roc_curve(layers_test, 
              [
               ("AE", "cae_score"),
               ("SNN", "ann_score_1"),
               ("DNN (2 layers)", "ann_score_2"),
               ("DNN (3 layers)", "ann_score_3"),
               ("DNN (4 layers)", "ann_score_4"),
               ("Simple test", "st_score"),
               ], (specificity_ann, sensitivity_ann))


# In[100]:


# Distribution of scores:
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, 1, 100)
plt.hist(layers_test[layers_test["score"] < 0]["ann_score_4"], bins=bins, alpha=0.5, label="Normalies")
plt.hist(layers_test[layers_test["score"] > 0]["ann_score_4"], bins=bins, alpha=0.5, label="Anomalies")
plt.title("Distribution of scores: DNN (302634 + 306125)")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
plt.show()


# In[101]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
#ax.set_xscale('log')
bins = np.linspace(0, 0.2, 100)
plt.hist(layers_test[layers_test["score"] < 0]["cae_score"], bins=bins, alpha=0.5, label="Normalies")
plt.hist(layers_test[layers_test["score"] > 0]["cae_score"], bins=bins, alpha=0.5, label="Anomalies")
#for i in [-1, +1]:
#    plt.hist(layers_test[(layers_test["score"] < 0) & (layers_test["wheel"] == int(i))]\
#             ["cae_score"], bins=bins, alpha=0.5, label="Normalies_Wheel_"+str(i))
plt.title("Distribution of scores: AE (302634 + 306125)")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.02, color='k', linestyle='dashed', linewidth=1)
plt.show()


# In[102]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
#ax.set_xscale('log')
bins = np.linspace(0, 0.2, 100)
plt.hist(layers_test[(layers_test["run"] == 306125) & (layers_test["score"] < 0)]["cae_score"],         bins=bins, alpha=0.5, label="306125 Normalies")
plt.hist(layers_test[(layers_test["run"] == 306125) & (layers_test["score"] > 0)]["cae_score"],         bins=bins, alpha=0.5, label="306125 Anomalies")
plt.title("Distribution of scores: AE (306125)")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.02, color='k', linestyle='dashed', linewidth=1)
plt.show()


# In[103]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
#ax.set_xscale('log')
bins = np.linspace(0, 0.2, 100)
plt.hist(layers_test[(layers_test["run"] == 302634) & (layers_test["score"] < 0)]["cae_score"],         bins=bins, alpha=0.5, label="302634 Normalies")
plt.hist(layers_test[(layers_test["run"] == 302634) & (layers_test["score"] > 0)]["cae_score"],         bins=bins, alpha=0.5, label="302634 Anomalies")
#for i in [-1, +1]:
#    plt.hist(layers_test[(layers_test["score"] < 0) & (layers_test["wheel"] == int(i))]\
#             ["cae_score"], bins=bins, alpha=0.5, label="Normalies_Wheel_"+str(i))
plt.title("Distribution of scores: AE (302634)")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.02, color='k', linestyle='dashed', linewidth=1)
plt.show()


# In[108]:


th_DNN = 0.1
y_pred = 2*(layers_test["ann_score_4"] > th_DNN)-1
layers_test["score"] = 2*(layers_test["score"] > th_DNN)-1

cnf_matrix = confusion_matrix(layers_test["score"], y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix DNN, with normalization')


# In[109]:


th_AE = 0.02
y_pred = 2*(layers_test["cae_score"] > th_AE)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix AE, with normalization')


# In[110]:


print "Number of FPs for the DNN:", len(layers_test[(layers_test["score"] == -1) &                                                    (layers_test["ann_score_4"] > th_DNN)])


# In[111]:


print "Number of FPs for the AE:", len(layers_test[(layers_test["score"] == -1) &                                                   (layers_test["cae_score"] > th_AE)])


# In[112]:


layers_test["name"] = ("W" + layers_test["wheel"].astype(str) + "_S" + layers_test["sector"].astype(str) +       "_St" + layers_test["station"].astype(str))


# Where are the FPs located?

# In[113]:


def count_fp(df, dis_nn, th, filt):
    df_temp = df[(layers_test["score"] == -1) & (layers_test[dis_nn] > th)].copy()
    df_temp = df_temp.groupby(['run', 'wheel', 'sector', 'station', 'name'])    .size().reset_index(name='counts')
    if filt:
        df_temp = df_temp[df_temp["counts"] > 1]
    #print df_temp
    return len(df_temp), df_temp

num_fp_ann, fp_ann = count_fp(layers_test, "ann_score_4", th_DNN, False)
num_fp_cae, fp_cae = count_fp(layers_test, "cae_score", th_AE, True)

print "Number of chambers with false positives DNN:", num_fp_ann
print "Number of chambers with false positives AE:", num_fp_cae
fp_cae.set_index("name",drop=True,inplace=True)
fp_ann.set_index("name",drop=True,inplace=True)


# In[114]:


fp_ann["counts"].plot(kind='bar', title ="False positives DNN", legend=True, fontsize=12)


# In[115]:


fp_cae["counts"].plot(kind='bar', title ="False positives AE", legend=True, fontsize=12)


# In[116]:


def get_matrix(df):
    x = np.zeros((5,12),dtype=int)
    for i in range(len(df)):
        a = int(5-df["station"].iloc[i])
        b = int(df["sector"].iloc[i]-1)
        x[a,b] = x[a,b] + 1
    return x


# In[117]:


def plot_scatter(df, run, wheel, ls_min, ls_max):
    run_s = "All"
    wheel_s = "All"
    ls_s = "All"
    temp = df.copy()
    if wheel != -3:
        temp = temp[(temp["wheel"] == wheel)]
        wheel_s = str(wheel)
    if run != -1:
        temp = temp[(temp["run"] == run)]
        run_s = str(run)
    if ((ls_min != -1) & (ls_max == -1)):
        temp = temp[(temp["averageLS"] >= ls_min)]
        ls_s = "> "+str(ls_min)
    elif ((ls_min == -1) & (ls_max != -1)):
        temp = temp[(temp["averageLS"] <= ls_max)]
        ls_s = "< "+str(ls_max)
    elif ((ls_min != -1) & (ls_max != -1)):
        temp = temp[(temp["averageLS"] >= ls_min) & (temp["averageLS"] <= ls_max)]
        ls_s = " ["+str(ls_min)+", "+str(ls_max)+"]"
    mat = get_matrix(temp)
    #print mat
    #print mat.sum()

    plt.figure()
    
    ax = plt.gca()
    ax.set_yticklabels(["1", "2", "3", "4", "5"])
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    plt.xlabel("Sector")
    plt.ylabel("Station")
    
    im = ax.imshow(mat, interpolation="nearest")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    for i in range(0,5):
        for j in range(0,12):
            text = ax.text(j, i, mat[i, j],            ha="center", va="center", color="w")
    
    plt.colorbar(im, cax=cax, ticks=[np.min(np.nan_to_num(mat)), np.max(np.nan_to_num(mat))])
    title = "Run: "+run_s+", Wheel: "+wheel_s+", LS: "+ls_s
    plt.title(title, loc="right")   
    plt.show()


# In[118]:


def plot_scatter_2(df, arg, wheel):
    wheel_s = "All"
    temp = df.copy()
    if wheel != -3:
        temp = temp[(temp["wheel"] == wheel)]
        wheel_s = str(wheel)
    else:
        return 1
    
    mat = []
    for i in [5, 4, 3, 2, 1]:
        rule = (temp.station == i)
        vec = list(temp[rule][arg].values)
        if (i == 5):
            tmp = list(temp[rule][arg].values)
            vec = [0, 0, 0, tmp[0], 0, 0, 0, 0, 0, tmp[1], 0, 0]
        mat.append(vec)
        #print i, temp[rule]["ratio"].values
    #print mat

    plt.figure()
    
    ax = plt.gca()
    ax.set_yticklabels(["1", "2", "3", "4", "5"])
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    plt.xlabel("Sector")
    plt.ylabel("Station")
    
    im = ax.imshow(mat, interpolation="nearest")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    for i in range(0,5):
        for j in range(0,12):
            text = ax.text(j, i, round(mat[i][j], 2),            ha="center", va="center", color="w")
    
    plt.colorbar(im, cax=cax, ticks=[np.min(np.nan_to_num(mat)), np.max(np.nan_to_num(mat))])
    title = "Wheel: "+wheel_s
    plt.title(title, loc="right")   
    plt.show()


# Decrease in rate per station and sector in each wheel:

# In[119]:


plot_scatter_2(ratio, "ratio", -2)
plot_scatter_2(ratio, "ratio", -1)
plot_scatter_2(ratio, "ratio", 0)
plot_scatter_2(ratio, "ratio", +1)
plot_scatter_2(ratio, "ratio", +2)


# Cross-section per station and sector in each wheel for the reference run:

# In[120]:


plot_scatter_2(ratio, "ref", -2)
plot_scatter_2(ratio, "ref", -1)
plot_scatter_2(ratio, "ref", 0)
plot_scatter_2(ratio, "ref", +1)
plot_scatter_2(ratio, "ref", +2)


# Cross-section per station and sector in each wheel for the anomalous run:

# In[121]:


plot_scatter_2(ratio, "anomal", -2)
plot_scatter_2(ratio, "anomal", -1)
plot_scatter_2(ratio, "anomal", 0)
plot_scatter_2(ratio, "anomal", +1)
plot_scatter_2(ratio, "anomal", +2)


# In[122]:


dis_nn = "cae_score"
rule = (layers_test["score"] == -1) & (layers_test[dis_nn] > th_AE)
plot_scatter(layers_test[rule], 302634, -3, 100, -1)


# In[123]:


plot_scatter(layers_test[rule], 302634, -2, 100, -1)
plot_scatter(layers_test[rule], 302634, -1, 100, -1)
plot_scatter(layers_test[rule], 302634, 0, 100, -1)
plot_scatter(layers_test[rule], 302634, +1, 100, -1)
plot_scatter(layers_test[rule], 302634, +2, 100, -1)


# Looking at how the recognized and unrecognized anomalies are distributed:

# In[124]:


print "ANOMALIES: NOT SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] < th_AE)
plot_scatter(layers_test[rule], 302634, -2, 100, -1)
print "ANOMALIES: SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] > th_AE)
plot_scatter(layers_test[rule], 302634, -2, 100, -1)


# In[125]:


print "ANOMALIES: NOT SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] < th_AE)
plot_scatter(layers_test[rule], 302634, -1, 100, -1)
print "ANOMALIES: SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] > th_AE)
plot_scatter(layers_test[rule], 302634, -1, 100, -1)


# In[126]:


print "ANOMALIES: NOT SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] < th_AE)
plot_scatter(layers_test[rule], 302634, 0, 100, -1)
print "ANOMALIES: SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] > th_AE)
plot_scatter(layers_test[rule], 302634, 0, 100, -1)


# In[127]:


print "ANOMALIES: NOT SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] < th_AE)
plot_scatter(layers_test[rule], 302634, +1, 100, -1)
print "ANOMALIES: SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] > th_AE)
plot_scatter(layers_test[rule], 302634, +1, 100, -1)


# In[128]:


print "ANOMALIES: NOT SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] < th_AE)
plot_scatter(layers_test[rule], 302634, +2, 100, -1)
print "ANOMALIES: SELECTED WITH AE"
rule = (layers_test["score"] == 1) & (layers_test[dis_nn] > th_AE)
plot_scatter(layers_test[rule], 302634, +2, 100, -1)


# Where are the FPs located in time?

# def deduceLS(data):
#     return data*10+5
# anomalies["averageLS"] = anomalies["group"].apply(deduceLS)

# In[129]:


threshold = th_AE
plotFpVsLs(302634, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",           threshold, True, boundaries[boundaries["run"] == 302634]["ls_end"])
#plotFpVsLs(302635, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",\
           #threshold, True, boundaries[boundaries["run"] == 302635]["ls_end"])


# Trying unsupervised algorithms:

# In[138]:


# fit the model
from sklearn.neighbors import LocalOutlierFactor
lofclf = LocalOutlierFactor(n_neighbors=1201,
                            contamination=0.15
                           )#It has to be a odd number
layers_test["lof_score"] = -lofclf.fit_predict(np.vstack(layers_test["content_scaled"].values))


# In[139]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(-1, +1, 3)
plt.hist(layers_test[layers_test["score"] < 0]["lof_score"], bins=bins, alpha=0.5, label="Normalies")
plt.hist(layers_test[layers_test["score"] > 0]["lof_score"], bins=bins, alpha=0.5, label="Anomalies")
plt.title("Distribution of scores: LOF (302634 + 306125)")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.show()


# In[140]:


threshold = 0.0
y_pred = 2*(layers_test["lof_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix LOF, with normalization')


# In[141]:


layers_test["averageLS"] = layers_test["group"].apply(deduceLS)
threshold = 0.0
plotFpVsLs(302634, 0, 0, 0, "Distribution of false positives: LOF, ", layers_test, "lof_score",           threshold, True, boundaries[boundaries["run"] == 302634]["ls_end"])
#plotFpVsLs(302635, 0, 0, 0, "Distribution of false positives: LOF, ", layers_test, "lof_score",\
            #threshold, True, boundaries[boundaries["run"] == 302635]["ls_end"])


# In[142]:


dis_nn = "lof_score"
rule = (layers_test["score"] == -1) & (layers_test[dis_nn] > 0)

plot_scatter(layers_test[rule], 302634, -2, 100, -1)
plot_scatter(layers_test[rule], 302634, -1, 100, -1)
plot_scatter(layers_test[rule], 302634, 0, 100, -1)
plot_scatter(layers_test[rule], 302634, +1, 100, -1)
plot_scatter(layers_test[rule], 302634, +2, 100, -1)


# ## K-Means clustering reduced

# In[143]:


factors = pd.DataFrame()

print("Loading SF")
path = "./factors.csv"
factors = factors.append(pd.read_csv(path,        names=["wheel", "sector", "station", "CS", "factor"]),        ignore_index=True)

print("Done.")


# In[172]:


print factors[(factors.wheel == +2) & (factors.sector == 10) & (factors.station == 3)]


# In[144]:


# 0.system, 1.wheel, 2.sector, 3.station, 4.lumi, 5.err, 6.rate, 7.err, 8.CS, 9.err
def removeItem(data):
    index = [0, 4, 5, 6, 7, 9] #leaving only position and CS
    #index = [0, 5, 7, 9] #removing system and errors
    temp = data.copy()
    temp = np.delete(temp, index)
    return temp

layers_test["content_red"] = layers_test["content"].apply(removeItem)
layers_test["content_red"] = layers_test["content_red"].apply(np.array)


# In[145]:


#Reindex because otherwise anomalies and normalies have the same labels/indexes in W+3_S4_MB3
layers_test.index = pd.RangeIndex(len(layers_test.index))

def scale_cs(df, factors):
    for i in [-2, -1, 0, +1, +2]:
        for j in range(1, 13):
             for k in range(1, 6):
                rule = (df.wheel == i) & (df.sector == j) & (df.station == k)
                rule_f = (factors.wheel == i) & (factors.sector == j) & (factors.station == k)
                indexes = df[rule].index
                indexes_f = (factors[rule_f].index)
                if ((len(indexes) == 0) | (len(indexes_f) == 0)):
                    continue
                #print factors.loc[indexes_f[0], "factor"]
                for m in indexes:
                    #print i, j, k, m, df.loc[m, "content_red"]
                    content =  df.loc[m, "content_red"].copy()
                    #print i, j, k, m, df.loc[m, "content_red"], content
                    content[3] = content[3]*factors.loc[indexes_f[0], "factor"]
                    #print i, j, k, m, df.loc[m, "content_red"], content
                    df.at[m, "content_red"] = content

scale_cs(layers_test, factors)


# In[146]:


#0.No rescaling
#1.RobustScaler
#2.MaxAbsScaler 
#3.QuantileTransformerUniform: NO
#4.QuantileTransformerNormal: NO
#5.Normalizer
#6.MinMaxScaler

scaler_type = 5
layers_test["content_red_scaled"] = layers_test["content_red"].apply(scale_data, args=[scaler_type])


# In[147]:


print layers_test["content_red"].iloc[100]
print layers_test["content_red_scaled"].iloc[100]


# In[148]:


def extract_cs(data):
    #print data[0][3]
    return data[3]

layers_test["CS_red"] = layers_test["content_red"].apply(extract_cs)


# In[149]:


def plot_scatter_3(df, arg, wheel):
    wheel_s = "All"
    temp = df.copy()
    if wheel != -3:
        temp = temp[(temp["wheel"] == wheel)]
        wheel_s = str(wheel)
    else:
        return 1
    
    temp = temp.groupby(['run', 'board', 'wheel', 'sector', 'system', 'station'])    [[arg]].mean().reset_index()
    #print temp
    
    mat = []
    for i in [5, 4, 3, 2, 1]:
        rule = (temp.station == i)
        temp2 = temp[rule].sort_values(["sector"], ascending=True)
        vec = list(temp2[arg].values)
        if (i == 5):
            tmp = list(temp2[arg].values)
            vec = [0, 0, 0, tmp[0], 0, 0, 0, 0, 0, tmp[1], 0, 0]
        mat.append(vec)
        #print i, temp[rule][arg].values
    #print mat

    plt.figure()
    
    ax = plt.gca()
    ax.set_yticklabels(["1", "2", "3", "4", "5"])
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    plt.xlabel("Sector")
    plt.ylabel("Station")
    
    im = ax.imshow(mat, interpolation="nearest")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    for i in range(0,5):
        for j in range(0,12):
            text = ax.text(j, i, round(mat[i][j], 2),            ha="center", va="center", color="w")
    
    plt.colorbar(im, cax=cax, ticks=[np.min(np.nan_to_num(mat)), np.max(np.nan_to_num(mat))])
    title = "Wheel: "+wheel_s
    plt.title(title, loc="right")   
    plt.show()


# In[174]:


rule = (layers_test.lumi > 6000)
plot_scatter_3(layers_test[layers_test.run == 306125], "CS_red", -2)
plot_scatter_3(layers_test[layers_test.run == 306125], "CS_red", -1)
plot_scatter_3(layers_test[layers_test.run == 306125], "CS_red", 0)
plot_scatter_3(layers_test[layers_test.run == 306125], "CS_red", +1)
plot_scatter_3(layers_test[(layers_test.run == 306125) & rule], "CS_red", +2)


# In[151]:


plot_scatter_3(layers_test[layers_test.run == 302634], "CS_red", -2)
plot_scatter_3(layers_test[layers_test.run == 302634], "CS_red", -1)
plot_scatter_3(layers_test[layers_test.run == 302634], "CS_red", 0)
plot_scatter_3(layers_test[layers_test.run == 302634], "CS_red", +1)
plot_scatter_3(layers_test[layers_test.run == 302634], "CS_red", +2)


# In[152]:


#n_cls = 150
#from sklearn import cluster, datasets
#k_means = cluster.KMeans(n_clusters=n_cls)
#distances = k_means.fit_transform(np.vstack(layers_test["content_red"].values))
#layers_test["kmeans_score"] = k_means.labels_


# In[153]:


# load the model from disk
from sklearn.externals import joblib
filename = './model_sktlearn/kmeans_red.sav'
k_means = joblib.load(filename)


# In[154]:


rule = (layers_test.run == 302634) & ((layers_test.ls < 40) | (layers_test.ls > 70))#& (layers_test.score == 1)
reduced_anom = layers_test[rule].copy()


# In[155]:


distances_test = k_means.transform(np.vstack(reduced_anom["content_red"].values))
minim = []
for i in range(0, len(distances_test)):
    #print  min(distances_test[i])
    minim.append(min(distances_test[i]))
reduced_anom["dist"] = minim


# In[156]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, 3.0, 100)
plt.hist(reduced_anom[reduced_anom["score"] < 0]["dist"], bins=bins,         alpha=0.5, label="Normal chambers")
plt.hist(reduced_anom[reduced_anom["score"] > 0]["dist"], bins=bins,         alpha=0.5, label="Anomalous chambers")
plt.axvline(0.4, color='k', linestyle='dashed', linewidth=1)
plt.title("K-Means average distance distribution (302634)")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Distance to the closest cluster')
plt.show()


# In[157]:


th_km = 0.4
y_pred = 2*(reduced_anom["dist"] > th_km)-1
cnf_matrix = confusion_matrix(reduced_anom["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix K-Means, with normalization')


# In[158]:


get_roc_curve(reduced_anom,[
                           #("Variance", "variance_score"),
                           #("IF", "if_score"),
                           #("SVM", "svm_score"),
                           #("AE", "cae_score"),
                           ("KMeans", "dist"),
                           #("DNN", "ann_score_4"),
                           ("Simple test", "st_score"),
                          ]
             )


# In[159]:


threshold = th_km
plotFpVsLs(302634, 0, 0, 0, "Distribution of false positives: KMeans clustering, ", reduced_anom, "dist",           threshold, True, boundaries[boundaries["run"] == 302634]["ls_end"])
#plotFpVsLs(302635, 0, 0, 0, "Distribution of false positives: LOF, ", layers_test, "lof_score",\
            #threshold, True, boundaries[boundaries["run"] == 302635]["ls_end"])


# In[160]:


dis_nn = "dist"
rule = (reduced_anom["score"] == -1) & (reduced_anom[dis_nn] > th_km)

plot_scatter(reduced_anom[rule], 302634, -2, 70, -1)
plot_scatter(reduced_anom[rule], 302634, -1, 70, -1)
plot_scatter(reduced_anom[rule], 302634, 0, 70, -1)
plot_scatter(reduced_anom[rule], 302634, +1, 70, -1)
plot_scatter(reduced_anom[rule], 302634, +2, 70, -1)


# In[161]:


print "ANOMALIES: NOT SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] < th_km)
plot_scatter(reduced_anom[rule], 302634, -2, 100, -1)
print "ANOMALIES: SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] > th_km)
plot_scatter(reduced_anom[rule], 302634, -2, 100, -1)


# In[162]:


print "ANOMALIES: NOT SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] < th_km)
plot_scatter(reduced_anom[rule], 302634, -1, 100, -1)
print "ANOMALIES: SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] > th_km)
plot_scatter(reduced_anom[rule], 302634, -1, 100, -1)


# In[163]:


print "ANOMALIES: NOT SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] < th_km)
plot_scatter(reduced_anom[rule], 302634, 0, 100, -1)
print "ANOMALIES: SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] > th_km)
plot_scatter(reduced_anom[rule], 302634, 0, 100, -1)


# In[164]:


print "ANOMALIES: NOT SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] < th_km)
plot_scatter(reduced_anom[rule], 302634, +1, 100, -1)
print "ANOMALIES: SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] > th_km)
plot_scatter(reduced_anom[rule], 302634, +1, 100, -1)


# In[165]:


print "ANOMALIES: NOT SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] < th_km)
plot_scatter(reduced_anom[rule], 302634, +2, 100, -1)
print "ANOMALIES: SELECTED WITH KMeans"
rule = (reduced_anom["score"] == 1) & (reduced_anom[dis_nn] > th_km)
plot_scatter(reduced_anom[rule], 302634, +2, 100, -1)


# ## LOF reduced

# In[166]:


rule1 = (layers_test.run == 302634) & ((layers_test.ls < 40) | (layers_test.ls > 70))
rule2 = (layers_test.run == 306125) & ((layers_test.ls < 200) | (layers_test.ls > 255)) &((layers_test.ls < 2265) | (layers_test.ls > 2315))
reduced_anom_2 = layers_test[rule1 | rule2].copy()


# In[167]:


# fit the model
lofclf = LocalOutlierFactor(n_neighbors=1201, contamination=0.2)#It has to be a odd number
reduced_anom_2["lof_score_red"] = -lofclf.fit_predict(np.vstack(reduced_anom_2["content_red"].values))


# In[168]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(-1, +1, 3)
plt.hist(reduced_anom_2[reduced_anom_2["score"] < 0]["lof_score_red"],         bins=bins, alpha=0.5, label="Normal chambers")
plt.hist(reduced_anom_2[reduced_anom_2["score"] > 0]["lof_score_red"],         bins=bins, alpha=0.5, label="Anomalous chambers")
plt.title("Distribution of scores: LOF")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.0, color='k', linestyle='dashed', linewidth=1)
plt.show()


# In[169]:


threshold = 0.0
y_pred = 2*(reduced_anom_2["lof_score_red"] > threshold)-1
cnf_matrix = confusion_matrix(reduced_anom_2["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix LOF, with normalization')


# In[170]:


dis_nn = "lof_score_red"
rule = (reduced_anom_2["score"] == -1) & (reduced_anom_2[dis_nn] > th_km)

plot_scatter(reduced_anom_2[rule], 302634, -2, 70, -1)
plot_scatter(reduced_anom_2[rule], 302634, -1, 70, -1)
plot_scatter(reduced_anom_2[rule], 302634, 0, 70, -1)
plot_scatter(reduced_anom_2[rule], 302634, +1, 70, -1)
plot_scatter(reduced_anom_2[rule], 302634, +2, 70, -1)


# In[171]:


reduced_anom_2["averageLS"] = reduced_anom_2["group"].apply(deduceLS)
threshold = 0.0
plotFpVsLs(302634, 0, 0, 0, "Distribution of false positives: LOF, ", reduced_anom_2, "lof_score_red",           threshold, True, boundaries[boundaries["run"] == 302634]["ls_end"])
#plotFpVsLs(302635, 0, 0, 0, "Distribution of false positives: LOF, ", layers_test, "lof_score",\
            #threshold, True, boundaries[boundaries["run"] == 302635]["ls_end"])


# In[ ]:




