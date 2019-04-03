#!/usr/bin/env python
# coding: utf-8

# ## TOC:
# * [Reading CVS files](#first-bullet)
# * [Luminosity section](#second-bullet)
# * [Trigger rate section](#third-bullet)
# * [Model training section](#fourth-bullet)

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
from keras.layers import Dense, Flatten, Reshape, Conv1D, MaxPooling1D, AveragePooling1D,UpSampling1D, InputLayer

from scipy import ndimage, misc

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn import cluster, datasets


# In[2]:


# Change presentation settings
#get_ipython().run_line_magic('matplotlib', 'inline')

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

def save_plot(plot):
    save_plot.counter +=1
    plot.savefig(str(save_plot.counter))
    plot.show()
save_plot.counter = 0

# In[3]:


#runs = [302634, 302635, 305814, 306121, 306122, 306125, 306126]
runs = [305814, 306121, 306122, 306125, 306126]
lumi_directory = data_directory = "./lumi"
rates_directory = "./rates"


# # Reading cvs files <a class="anchor" id="first-bullet"></a>

# Reading instantaneous luminosities from the cvs file produced with brilcalc and saving into a pandas dataframe:

# In[4]:


df_rates = pd.DataFrame()
int_lumi2 = pd.DataFrame()
for run in runs:
    print(("Loading %s" % run))
    path = "%s/lumi_%s.csv" % (lumi_directory, run)
    int_lumi2 = int_lumi2.append(pd.read_csv(path,
        names=["runfill", "ls", "time", "beamstatus", "energy", "delivered",\
               "recorded", "avgpu", "source"]), 
        ignore_index=True)
    path = "%s/dt_rates_%s.csv" % (rates_directory, run)
    df_rates = df_rates.append(pd.read_csv(path, 
        names=["run", "time", "board", "RPC1", "RPC2", "RPC3", "RPC4",\
               "DT1", "DT2", "DT3", "DT4", "DT5"]), 
        ignore_index=True)
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
    print(i, start.iloc[0], start.iloc[-1], start_ls.iloc[0], start_ls.iloc[-1], nLS)
    boundaries = boundaries.append({"run": i, "start": start.iloc[0], "end": start.iloc[-1], 
                                   "ls_start": start_ls.iloc[0], "ls_end": start_ls.iloc[-1],\
                                    "nLS": nLS}, ignore_index = True)


# In[11]:


boundaries = boundaries.sort_values('run')
boundaries = boundaries.reset_index()


# Reindexing the dataframe after removing some lines:

# In[12]:


int_lumi2.index = pd.RangeIndex(len(int_lumi2.index))


# In[13]:


print(len(int_lumi2.index))


# Filling end time column:

# In[14]:


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


# Selecting only beam status STABLE BEAM:

# In[15]:


#print int_lumi2["beamstatus"]
int_lumi2 = int_lumi2[int_lumi2["beamstatus"] == "STABLE BEAMS"]


# In[16]:


int_lumi2.to_csv("int_lumi2.csv", sep='\t')


# Plotting the instantaneous luminosities:

# In[17]:


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
    save_plot(plt);


# In[18]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 306121]["time"], 
               int_lumi2[int_lumi2["run"] == 306121]["delivered"], 
               int_lumi2[int_lumi2["run"] == 306121]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               (boundaries["run"].iloc[1], int_lumi2["fill"].iloc[0])))


# In[19]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 306122]["time"], 
               int_lumi2[int_lumi2["run"] == 306122]["delivered"], 
               int_lumi2[int_lumi2["run"] == 306122]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               (boundaries["run"].iloc[2], int_lumi2["fill"].iloc[0])))


# In[20]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 306125]["time"], 
               int_lumi2[int_lumi2["run"] == 306125]["delivered"], 
               int_lumi2[int_lumi2["run"] == 306125]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               (boundaries["run"].iloc[3], int_lumi2["fill"].iloc[0])))


# In[21]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 306126]["time"], 
               int_lumi2[int_lumi2["run"] == 306126]["delivered"], 
               int_lumi2[int_lumi2["run"] == 306126]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               (boundaries["run"].iloc[4], int_lumi2["fill"].iloc[0])))


# In[22]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 305814]["time"], 
               int_lumi2[int_lumi2["run"] == 305814]["delivered"], 
               int_lumi2[int_lumi2["run"] == 305814]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               (boundaries["run"].iloc[0], int_lumi2["fill"].iloc[0])))


# # Trigger rate section <a class="anchor" id="third-bullet"></a>

# Converting columns to proper data types:

# In[23]:


df_rates["time"] = pd.to_datetime(df_rates["time"])
df_rates["run"] = df_rates["run"].astype('int')
#print df_rates["time"]


# Splitting, converting and adding new columns:

# In[24]:


df_rates['wheel'], df_rates['sector'] = df_rates['board'].str.split('_', 1).str
df_rates["wheel"] = df_rates["wheel"].astype(str)
df_rates["sector"] = df_rates["sector"].astype(str)


# In[25]:


df_rates["wheel"].replace(regex=True,inplace=True,to_replace=r'YB',value=r'')
df_rates["sector"].replace(regex=True,inplace=True,to_replace=r'S',value=r'')
df_rates["wheel"] = df_rates["wheel"].astype('int')
df_rates["sector"] = df_rates["sector"].astype('int')
df_rates["ls"] = -1
df_rates["lumi"] = -1.0
df_rates["score"] = -1
df_rates.to_csv("df_rates.csv", sep='\t')


# Plotting the rate coming from one of the stations:

# In[26]:


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
    save_plot(plt);

rule = (df_rates.run != 305814)
plot_rate_vs_time(df_rates[rule],                  "time", "DT1", "YB+1_S4", "Rates for Runs / Fill / Board: %s / %s / %s" % 
                  (str(boundaries["run"].iloc[1])+"-"+str(boundaries["run"].iloc[3]), 
                   int_lumi2["fill"].iloc[2], "YB+1_S4"))


# Associating a LS and an instantaneous luminosity to each rate:

# In[27]:


#Just a backup copy
df_rates_backup = df_rates.copy()


# In[28]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# Removing the measurements taken before and after the start and end time reported by the brilcalc output. All the 60 boards are measured at the same time. In order to speed-up the association, we take just one board, the first one. This reduces the dataframe and the time needed to go though it by a factor of 60.

# In[29]:

df_rates_tmp = pd.DataFrame()
for run in runs :
    df_rates_per_run = df_rates.copy()
    run_boundaries = boundaries[boundaries["run"]==run]
    time0 = run_boundaries["start"].iloc[0]
    timeF = run_boundaries["end"].iloc[-1]
    df_rates_per_run = df_rates_per_run[(df_rates_per_run.run==run) & (df_rates_per_run.time >= time0) & (df_rates_per_run.time <= timeF)]
    frames = [df_rates_tmp, df_rates_per_run]
    df_rates_tmp = pd.concat(frames)

#df_rates_tmp.to_csv("df_rates_tmp.csv",  sep='\t')
rule = df_rates_tmp.duplicated(subset=["time"])
count = (rule == False).sum()
print("Duplicates:", rule.sum())
df_rates_noduplicates = df_rates_tmp[rule == False]

# In[30]:


print(len(df_rates_noduplicates))


# Assigning the LS and the inst. lumi. to the measurements for the selected board:

# In[31]:


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
                    print(j)
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

# In[32]:


df_rates_noduplicates = df_rates_noduplicates[df_rates_noduplicates["ls"] > 0]
print(len(df_rates_noduplicates))


# Save in a csv file:

# In[33]:


df_rates.to_csv("df_rates.csv", sep='\t')
df_rates_noduplicates.to_csv("df_rates_nodup.csv", sep='\t')


# In[34]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# Assign the LS and the inst. lumi. to all the 60 boards for each time:

# In[35]:


def assignLS_ext(df1, df2):
    temp = df1.copy()
    indexes = []
    for index in df2.index:
        if index%10000 == 0:
            print(index)
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


# In[36]:


df_rates = temp.copy()
#print df_rates[df_rates.ls <= 0]


# Removing measurements without LS assignment:

# In[37]:


df_rates_backup = df_rates.copy()
df_rates = df_rates[df_rates.ls > 0]
#print df_rates["ls"]


# In[38]:


#print df_rates[df_rates.ls <= 0]
df_rates.to_csv("df_rates.csv", sep='\t')


# Averaging the rates associated to the same LS:

# In[39]:


df_boards = df_rates.copy()
df_boards = df_boards.groupby(['board']).size().reset_index(name='counts')
print(len(df_boards))
#print df_boards


# Too slow to use all the measurements. Averaging over 10 LS:

# In[40]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# In[41]:


bunch = 10
def assignGroup(data, div = bunch):
    res = int(data/div)
    #print data, res
    return res

df_rates["group"] = df_rates["ls"]
df_rates["group"] = df_rates["group"].apply(assignGroup)


# In[42]:


#print df_rates["group"]


# In[43]:


temp1 = df_rates.groupby(['run', 'group', 'board', 'wheel', 'sector'])[["ls", "lumi", "RPC1", "RPC2", "RPC3", "RPC4",  "DT1", "DT2", "DT3", "DT4", "DT5"]].mean().reset_index()

temp2 = df_rates.groupby(['run', 'group', 'board', 'wheel', 'sector'])[["lumi", "RPC1", "RPC2", "RPC3", "RPC4",  "DT1", "DT2", "DT3", "DT4", "DT5"]].std().reset_index()

temp3 = df_rates.groupby(['run', 'group', 'board', 'wheel', 'sector'])[["lumi", "RPC1", "RPC2", "RPC3", "RPC4",  "DT1", "DT2", "DT3", "DT4", "DT5"]].size().reset_index(name='counts')

temp2 = temp2.rename(index=str, columns={"lumi": "errLumi", "RPC1": "errRPC1", "RPC2": "errRPC2",                                         "RPC3": "errRPC3", "RPC4": "errRPC4", "DT1": "errDT1",                                         "DT2": "errDT2", "DT3": "errDT3", "DT4":                                         "errDT4", "DT5": "errDT5"})

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

# In[44]:


import math
def applySqrt(data):
    return math.sqrt(data)

df_rates["counts"] = df_rates["counts"].apply(applySqrt)

for i in list(df_rates):
    if "err" in i:
        #print i
        df_rates[i] = df_rates[i]/df_rates["counts"]


# In[45]:


#print df_rates


# Check for null or NaN values:

# In[46]:


print(df_rates.isnull().values.any())
null_columns=df_rates.columns[df_rates.isnull().any()]
print((df_rates[df_rates.isnull().any(axis=1)][null_columns].head()))
#df_rates = df_rates.fillna(0)
#print(df_rates[df_rates.isnull().any(axis=1)][null_columns].head())


# In[47]:


#Another backup
#df_rates_backup = df_rates.copy()
df_rates.to_csv("df_rates.csv", sep='\t')


# In[48]:


#Restore backup
#df_rates = df_rates_backup.copy()


# In[49]:


print(len(df_rates))


# Uncomment to check just one case case:

# In[50]:


#for index, row in df_rates.iterrows():
    #if row["board"] == "YB0_S1":
        #print "Index:", index,", Run:", row["run"],", Board: ",row["board"],",\
        #LS: ",row["ls"],", Rate: ",row["DT1"],", Error: ",row["errDT1"]


# Plotting the result:

# In[51]:


def plot_rate_vs_ls(df, run, x_val, y_val, z_val, x_err, y_err, title_x, title_y, title, opt):
    df_temp = df.copy()
    rule = ((df_temp["board"] == z_val) & (df_temp["run"] == run))
    df_temp = df_temp[rule]
    fig, ax = plt.subplots()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    fig.autofmt_xdate()
    plt.plot(df_temp[x_val], df_temp[y_val], opt)
    plt.legend(loc="best")
    plt.errorbar(df_temp[x_val], df_temp[y_val], xerr=x_err, yerr=df_temp[y_err], fmt='ro', ecolor='r')
    plt.title(title)
    save_plot(plt);

title = "Rates for Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+" / "+str(boundaries["run"].iloc[3])+" / YB+1_S4"

plot_rate_vs_ls(df_rates, 306125, "lumi", "DT1", "YB+1_S4", 0, "errDT1",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro")
plot_rate_vs_ls(df_rates, 306125, "lumi", "DT2", "YB+1_S4", 0, "errDT2",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro")
plot_rate_vs_ls(df_rates, 306125, "lumi", "DT3", "YB+1_S4", 0, "errDT3",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro")
plot_rate_vs_ls(df_rates, 306125, "lumi", "DT4", "YB+1_S4", 0, "errDT4",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro")
plot_rate_vs_ls(df_rates, 306125, "lumi", "DT5", "YB+1_S4", 0, "errDT5",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, "ro")


# In[52]:


title = "Rates for Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+" / "+str(boundaries["run"].iloc[0])+" / YB+1_S4"
plot_rate_vs_ls(df_rates[(df_rates["ls"] > 100) & (df_rates["ls"] < 205)], 305814,                "ls", "DT3", "YB-2_S2", 0,                "errDT3", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", "Rate [Hz]", title, "ro")


# Create a new dataframe with the input features already organized in a numpy array:

# In[53]:


print(len(df_rates))
#Uncomment to include RPC
#algos = ['RPC1', 'RPC2', 'RPC3', 'RPC4', 'DT1', 'DT2', 'DT3', 'DT4', 'DT5']
algos = ['DT1', 'DT2', 'DT3', 'DT4', 'DT5']
df_rates_new_2 = pd.DataFrame(columns=['run', 'group', 'board', 'wheel', 'sector', 'ls',                                     'lumi', 'errLumi', 'rate', 'err', 'system', 'station'])

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
    df_rates_new_2 = pd.concat([df_rates_new_2, temp], ignore_index=True)

print(len(df_rates_new_2))


# Adding the lumi/rate ratio:

# In[54]:


df_rates_new_2["CS"] = -1
df_rates_new_2["errCS"] = -1

df_rates_new_2["CS"] = df_rates_new_2["rate"]/df_rates_new_2["lumi"]
print("Number of NaN's in CS before:")
print(len(df_rates_new_2["CS"][df_rates_new_2["CS"].isnull() == True]))
print("Number of Inf's in CS before:")
print(len(df_rates_new_2["CS"][np.isinf(df_rates_new_2["CS"])]))

df_rates_new_2["CS"] = df_rates_new_2["CS"].replace([np.inf, -np.inf], np.nan)
df_rates_new_2["CS"] = df_rates_new_2["CS"].fillna(-1)

print("Number of NaN's in CS after:")
print(len(df_rates_new_2["CS"][df_rates_new_2["CS"].isnull() == True]))
print("Number of Inf's in CS after:")
print(len(df_rates_new_2["CS"][np.isinf(df_rates_new_2["CS"])]))

df_rates_new_2["errCS"] = (1/df_rates_new_2["lumi"])*np.sqrt(df_rates_new_2["err"]**2 + df_rates_new_2["CS"]**2 * df_rates_new_2["errLumi"]**2)

print("Number of NaN's in errCS before:")
print(len(df_rates_new_2["errCS"][df_rates_new_2["errCS"].isnull() == True]))
print("Number of Inf's in errCS before:")
print(len(df_rates_new_2["errCS"][np.isinf(df_rates_new_2["errCS"])]))

df_rates_new_2["errCS"] = df_rates_new_2["errCS"].replace([np.inf, -np.inf], np.nan)
df_rates_new_2["errCS"] = df_rates_new_2["errCS"].fillna(-1)

print("Number of NaN's in errCS after:")
print(len(df_rates_new_2["errCS"][df_rates_new_2["errCS"].isnull() == True]))
print("Number of Inf's in errCS after:")
print(len(df_rates_new_2["errCS"][np.isinf(df_rates_new_2["errCS"])]))


# In[55]:


array = df_rates_new_2.as_matrix(columns=['system', 'wheel', 'sector', 'station',                                        'lumi', 'errLumi', 'rate', 'err',                                        'CS', 'errCS'])


# In[56]:


df_rates_new_2["content"] = np.empty((len(df_rates_new_2), 0)).tolist()
for index, rows in df_rates_new_2.iterrows():
    #print index, array[index]
    df_rates_new_2.at[index, "content"] = array[index]
df_rates_new_2["score"] = -1


# Actually station 5 exists only for sector 4 and 10. Removing the rows for all the other sectors:

# In[57]:


rule = ((df_rates_new_2["station"] == 5) & ((df_rates_new_2["sector"] != 4)                                            & (df_rates_new_2["sector"] != 10)))
df_rates_new_2 = df_rates_new_2[rule == False]


# In[58]:


def plot_rate_vs_ls_2(df1, df2, x_val, y_val, x_err, y_err, title_x, title_y, title, opt):
    fig, ax = plt.subplots()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    fig.autofmt_xdate()
    plt.errorbar(df1[x_val], df1[y_val], xerr=x_err, yerr=df1[y_err], fmt='ro', ecolor='r')
    num = y_val
    num = num.replace("DT", "")
    tmp = df2[df2.station == int(num)]
    plt.errorbar(tmp[x_val], tmp["rate"], xerr=x_err, yerr=tmp["err"], fmt='b+', ecolor='b')
    plt.title(title)
    plt.legend(loc="best")
    save_plot(plt);

title = "Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+" / "+str(boundaries["run"].iloc[3])+" / YB+1_S4"

rule_1 = ((df_rates["wheel"] == 1) & (df_rates["sector"] == 4) & (df_rates["run"] == 306125))
rule_2 = ((df_rates_new_2["wheel"] == 1) & (df_rates_new_2["sector"] == 4)          & (df_rates_new_2["run"] == 306125))

temp1 = df_rates[rule_1]
temp2 = df_rates_new_2[rule_2]

plot_rate_vs_ls_2(temp1, temp2, "lumi", "DT1", 0, "errDT1", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "lumi", "DT2", 0, "errDT2", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "lumi", "DT3", 0, "errDT3", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "lumi", "DT4", 0, "errDT4", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "lumi", "DT5", 0, "errDT5", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")


# In[400]:


title = "Fill/Run/Board: "+str(int_lumi2["fill"].iloc[0])+" / "+str(boundaries["run"].iloc[3])+" / YB-2_S4"

plot_rate_vs_ls(df_rates_new_2, 306125, "lumi", "CS", "YB-2_S4", 0, "errCS",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                r"Cross-Section [$\times10^{-30}$ cm$^{2}$]", title, "ro")


# In[60]:


def plot_scatter_2(df, arg, wheel):
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
    save_plot(plt)


# In[396]:


rule = (df_rates_new_2.lumi > 6000)
plot_scatter_2(df_rates_new_2[(df_rates_new_2.run == 306125) & rule], "CS", -2)
plot_scatter_2(df_rates_new_2[(df_rates_new_2.run == 306125) & rule], "CS", -1)
plot_scatter_2(df_rates_new_2[(df_rates_new_2.run == 306125) & rule], "CS", 0)
plot_scatter_2(df_rates_new_2[(df_rates_new_2.run == 306125) & rule], "CS", +1)
plot_scatter_2(df_rates_new_2[(df_rates_new_2.run == 306125) & rule], "CS", +2)


# In[62]:


print(df_rates_new_2.columns)
rule = (df_rates_new_2.run == 306125)
plt.plot(df_rates_new_2[rule]["group"]*10+5, df_rates_new_2[rule]["lumi"], 'bo')
plt.axvline(200, color='k', linestyle='dashed', linewidth=1)
plt.axvline(255, color='k', linestyle='dashed', linewidth=1)
plt.axvline(2265, color='k', linestyle='dashed', linewidth=1)
plt.axvline(2315, color='k', linestyle='dashed', linewidth=1)


# In[63]:


print(df_rates_new_2.columns)
rule = (df_rates_new_2.run == 306126)
plt.plot(df_rates_new_2[rule]["group"]*10+5, df_rates_new_2[rule]["lumi"], 'bo')
plt.axvline(65, color='k', linestyle='dashed', linewidth=1)
plt.axvline(265, color='k', linestyle='dashed', linewidth=1)
plt.axvline(445, color='k', linestyle='dashed', linewidth=1)
plt.axvline(475, color='k', linestyle='dashed', linewidth=1)


# # Model training section <a class="anchor" id="fourth-bullet"></a>

# ## Creating train and test samples

# In[64]:


df_rates_new = df_rates_new_2.copy()


# In[65]:


print(df_rates_new_2["content"].iloc[1])


# In[66]:


#To be run just one time!!!!!!
df_rates_new_2["content_in"] = df_rates_new_2["content"].copy()
def change_data(data):
    temp = data.copy()
    temp[2] = temp[2] - 6
    temp[3] = temp[3] - 3
    return temp

df_rates_new_2["content"] = df_rates_new_2["content"].apply(change_data)


# In[67]:


print(df_rates_new_2["content"].iloc[1])


# In[68]:


anomalies = df_rates_new_2.copy()


# In[69]:


normalies = anomalies.copy()


# In[70]:


print(len(normalies), len(anomalies))


# In[71]:


rule = (normalies["wheel"] == -1) & (normalies["sector"] == 3) & (normalies["station"] == 3) 
print("Normal chimney:")
print(normalies[rule]["content"].iloc[0])

rule = (normalies["wheel"] == 1) & (normalies["sector"] == 4) & (normalies["station"] == 3)
print("Anomalous chimney:")
print(normalies[rule]["content"].iloc[0])


# In[72]:


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


# In[73]:


#rule = (anomalies["wheel"] == 1) & (anomalies["sector"] == 4) & (anomalies["station"] == 3)
#print anomalies[rule]["score"]


# Forcing the rate to the one of the symmetric chimney chamber:

# In[74]:


def assignRate(df):
    rule1 = (df["wheel"] == 1) & (df["sector"] == 4) & (df["station"] == 3)# not good
    indexes1 = df[rule1].index
    #print indexes
    for i in indexes1:
        group = df.loc[i]["group"]
        run = df.loc[i]["run"]
        rule2 = (df["wheel"] == -1) & (df["sector"] == 3) & (df["station"] == 3) &            (df["group"] == group) & (df["run"] == run)
        indexes2 = (df[rule2].index) #it should contain one index
        #print i, index, time
        if(len(indexes2) > 0):
            j = indexes2[0]
            array = df.loc[j]["content"]
            rate = array[6] #good rate from the symmetric sector
            err = array[7] #uncertainty on the good rate from the symmetric sector
            CS = array[8]
            errCS = array[9]
            content_orig = df.loc[i]["content"]
            content = list(content_orig)
            content[6] = rate
            content[7] = err
            content[8] = CS
            content[9] = errCS
            df.loc[i, "rate"] = rate
            df.loc[i, "err"] = err
            df.loc[i, "CS"] = CS
            df.loc[i, "errCS"] = errCS
            df.at[i, "content"] = content
            #print i, j, time, rate, content_orig, content
        else:
            content_orig = df.loc[i]["content"]
            content = list(content_orig)
            content[6] = -1
            content[7] = -1
            content[8] = -1
            content[9] = -1
            df.at[i, "content"] = content
            #print i, j, time, rate, content_orig, content

assignRate(normalies)


# Check that the change affects only normalies:

# In[75]:


rule = (normalies["wheel"] == -1) & (normalies["sector"] == 3) & (normalies["station"] == 3)
print("Normal chimney:")
print(normalies[rule]["content"].iloc[0])

rule = (normalies["wheel"] == 1) & (normalies["sector"] == 4) & (normalies["station"] == 3)
print("Anomalous chimney:")
print(normalies[rule]["content"].iloc[0])


# In[76]:


rule = (anomalies["wheel"] == -1) & (anomalies["sector"] == 3) & (anomalies["station"] == 3)
print("Normal chimney:")
print(anomalies[rule]["content"].iloc[0])

rule = (anomalies["wheel"] == 1) & (anomalies["sector"] == 4) & (anomalies["station"] == 3)
print("Anomalous chimney:")
print(anomalies[rule]["content"].iloc[0])


# In[401]:


rule = (normalies.lumi > 6000)
plot_scatter_2(normalies[(normalies.run == 306125) & rule], "CS", -2)
plot_scatter_2(normalies[(normalies.run == 306125) & rule], "CS", -1)
plot_scatter_2(normalies[(normalies.run == 306125) & rule], "CS", 0)
plot_scatter_2(normalies[(normalies.run == 306125) & rule], "CS", +1)
plot_scatter_2(normalies[(normalies.run == 306125) & rule], "CS", +2)


# In[78]:


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


# In[79]:


anomalies["content"] = anomalies["content"].apply(np.array)
anomalies["content_scaled"] = anomalies["content"].apply(scale_data)

normalies["content"] = normalies["content"].apply(np.array)
normalies["content_scaled"] = normalies["content"].apply(scale_data)


# In[80]:


#print anomalies["content_scaled"]
#print normalies["content_scaled"]


# In[81]:


# Set a random seed to reproduce the results
rng = np.random.RandomState(0)
anomalies = anomalies[(anomalies.score == 1) & (anomalies.run != 305814)]
normalies = normalies[(normalies.score == -1) & (normalies.run != 305814)]
print(("%s faults and %s good samples. In total: %s." %
      (len(anomalies), len(normalies), len(anomalies) + len(normalies))))

anomalies_train, anomalies_test = train_test_split(anomalies, test_size = 0.2, random_state=rng)
normalies_train, normalies_test = train_test_split(normalies, test_size = 0.2, random_state=rng)

neural_anomalies_train, neural_anomalies_val = train_test_split(anomalies_train,                                                                test_size = 0.2, random_state=rng)
neural_normalies_train, neural_normalies_val = train_test_split(normalies_train,                                                                test_size = 0.2, random_state=rng)

layers_train = pd.concat([anomalies_train, normalies_train])
layers_test = pd.concat([anomalies_test, normalies_test])

neural_train = pd.concat([neural_anomalies_train, neural_normalies_train])
neural_val = pd.concat([neural_anomalies_val, neural_normalies_val])


# In[82]:


print(("Number of anomalies in the train set: %s" % len(anomalies_train)))
print(("Number of normal in the train set: %s" % len(normalies_train)))
print(("Number of anomalies in the test set: %s" % len(anomalies_test)))
print(("Number of normal in the test set: %s" % len(normalies_test)))


# In[83]:


def score_to_array(score):
    if score == -1:
        return np.asarray([1, 0]) #Normaly
    return np.asarray([0, 1]) #Anomaly

def nn_generate_input():  
    return (np.array(np.concatenate(neural_train.content_scaled.values)).reshape(-1, 10),
            np.concatenate(neural_train["score"].apply(score_to_array).values).reshape(-1, 2),
            np.array(np.concatenate(neural_val.content_scaled.values)).reshape(-1, 10),
            np.concatenate(neural_val["score"].apply(score_to_array).values).reshape(-1, 2),
            np.array(np.concatenate(layers_test.content_scaled.values)).reshape(-1, 10))

(train_x, train_y, val_x, val_y, test_x) = nn_generate_input()


# In[84]:


def cae_generate_input():
    return np.array(np.concatenate(normalies.content_scaled.values)).reshape(-1, 10)

train_cae = cae_generate_input()


# In[85]:


from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import class_weight

cw = class_weight.compute_class_weight("balanced",
                                       np.unique(np.argmax(train_y, axis=1)),
                                       np.argmax(train_y, axis=1))
cw = {0: cw[0], 1: cw[1]}
print(cw)


# ## Simple test

# In[297]:


cross_sections = pd.DataFrame(columns=["wheel", "sector", "station", "CS", "errCS"])
def extract_cs(df, cross_sections):
    for i in [-2, -1, 0, +1, +2]:
        for j in range(1, 13):
            range_ = list(range(1, 5))
            if ((j == 4) | (j == 10)):
                range_ = list(range(1, 6))
            for k in range_:
                rule = ((df.wheel  == i) & (df.sector == j) &                        (df.station == k))
                val1 = df[rule]["lumi"]
                val2 = df[rule]["CS"]
                fit, cov = np.polyfit(val1, val2, 0, cov=True)
                err = np.sqrt(np.diag(cov))
                cross_sections = cross_sections.append({"wheel": i, "sector": j, "station": k,                                                        "CS": float(fit), "errCS": float(err)},                                                       ignore_index = True)
    return cross_sections


# In[298]:


cross_sections = extract_cs(normalies_train[normalies_train.lumi > 6000], cross_sections)


# In[314]:


print(cross_sections[(cross_sections.wheel == +2) & (cross_sections.sector == 10) &                     (cross_sections.station == 3)]["CS"])
print(cross_sections[(cross_sections.wheel == +2) & (cross_sections.sector == 10) &                     (cross_sections.station == 4)]["CS"])


# In[299]:


def simple_test(cross_sections, df, coeff):
    for i in df.index:
        wheel = df.wheel.loc[i]
        sector = df.sector.loc[i]
        station = df.station.loc[i]
        cs = df.CS.loc[i]
        err_cs = df.errCS.loc[i]
        #print cs
        rule = ((cross_sections.wheel == wheel) & (cross_sections.sector == sector) &                (cross_sections.station == station))
        cs_norm = cross_sections[rule].CS.iloc[0]
        err_cs_norm = cross_sections[rule].errCS.iloc[0]
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


# In[300]:


layers_test["st_score"] = 0
layers_test.index = pd.RangeIndex(len(layers_test.index))
simple_test(cross_sections, layers_test, 0.05)


# In[366]:


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
save_plot(plt)


# In[367]:


th_km = 0.0
y_pred = 2*(layers_test["st_score"] > th_km)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix Simple Test, with normalization')


# ## NN architectures

# Defining NN structure:

# In[93]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
K.set_session(sess)


# In[94]:


dim = 10
def neural_network_1():
    model = Sequential()
    model.add(Reshape((dim, 1), input_shape=(dim,), name='input_ann'))
    model.add(Flatten(name="flatten_ann"))
    model.add(Dense(32, activation='relu', name="dense_ann"))
    model.add(Dense(2, activation='softmax', name='output_ann'))
    return model

def neural_network_2():
    model = Sequential()
    model.add(Reshape((dim, 1), input_shape=(dim,), name='input_ann'))
    model.add(Flatten(name="flatten_ann"))
    model.add(Dense(32, activation='relu', name="dense_ann"))
    model.add(Dense(32, activation='relu', name="dense_ann2"))
    model.add(Dense(2, activation='softmax', name='output_ann'))
    return model

def neural_network_3():
    model = Sequential()
    model.add(Reshape((dim, 1), input_shape=(dim,), name='input_ann'))
    model.add(Flatten(name="flatten_ann"))
    model.add(Dense(32, activation='relu', name="dense_ann"))
    model.add(Dense(32, activation='relu', name="dense_ann2"))
    model.add(Dense(32, activation='relu', name="dense_ann3"))
    model.add(Dense(2, activation='softmax', name='output_ann'))
    return model

def neural_network_4():
    model = Sequential()
    model.add(Reshape((dim, 1), input_shape=(dim,), name='input_ann'))
    model.add(Flatten(name="flatten_ann"))
    model.add(Dense(32, activation='relu', name="dense_ann"))
    model.add(Dense(32, activation='relu', name="dense_ann2"))
    model.add(Dense(32, activation='relu', name="dense_ann3"))
    model.add(Dense(32, activation='relu', name="dense_ann4"))
    model.add(Dense(2, activation='softmax', name='output_ann'))
    return model

def autoencoder():
    from keras.layers import Input, Dense
    from keras.models import Model
    input_ = Input(shape=(10,))
    encoded = Dense(10, activation='relu')(input_)
    encoded = Dense(9, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded)
    encoded = Dense(7, activation='relu')(encoded)
    encoded = Dense(6, activation='relu')(encoded)
    
    decoded = Dense(6, activation='relu')(encoded)
    decoded = Dense(7, activation='relu')(decoded)
    decoded = Dense(8, activation='relu')(decoded)
    decoded = Dense(9, activation='relu')(decoded)
    decoded = Dense(10, activation='sigmoid')(decoded)

    autoencoder = Model(input_, decoded)
    return autoencoder


# In[95]:


ann_1 = neural_network_1()
ann_2 = neural_network_2()
ann_3 = neural_network_3()
ann_4 = neural_network_4()
#print("Neural Network Architecture:")
#ann_1.summary()
#ann_2.summary()
#ann_3.summary()
#ann_4.summary()


# Training the NN:

# In[96]:


# This may take some time...

def train_nn(model, x, y, batch_size, loss, name, validation_data=None, 
             validation_split=0.0, class_weight=None):

    model.compile(loss=loss, optimizer='Nadam')

    early_stopper = EarlyStopping(monitor="val_loss",
                                  patience=32,
                                  verbose=True,
                                  mode="auto")
    
    checkpoint_callback = ModelCheckpoint(("./model_keras/%s.h5" % name),
                                          monitor="val_loss",
                                          verbose=False,
                                          save_best_only=True,
                                          mode="min")
    return model.fit(x, y,
                     batch_size=batch_size,
                     epochs=8192,
                     verbose=False,
                     class_weight=class_weight,
                     shuffle=True,
                     validation_split=validation_split,
                     validation_data=validation_data,
                     callbacks=[early_stopper, checkpoint_callback])


# In[696]:


history_ann_1 = train_nn(ann_1,
                       train_x,
                       train_y,
                       len(train_x),
                       keras.losses.categorical_crossentropy,
                       "ann_1",
                       validation_data=(val_x, val_y),
                       class_weight=cw)


# In[697]:


history_ann_2 = train_nn(ann_2,
                       train_x,
                       train_y,
                       len(train_x),
                       keras.losses.categorical_crossentropy,
                       "ann_2",
                       validation_data=(val_x, val_y),
                       class_weight=cw)


# In[698]:


history_ann_3 = train_nn(ann_3,
                       train_x,
                       train_y,
                       len(train_x),
                       keras.losses.categorical_crossentropy,
                       "ann_3",
                       validation_data=(val_x, val_y),
                       class_weight=cw)


# In[699]:


history_ann_4 = train_nn(ann_4,
                       train_x,
                       train_y,
                       len(train_x),
                       keras.losses.categorical_crossentropy,
                       "ann_4",
                       validation_data=(val_x, val_y),
                       class_weight=cw)


# In[ ]:


def plot_training_loss(data, title):
    """Plots the training and validation loss"""
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.plot(data["loss"])
    plt.plot(data["val_loss"])
    plt.legend(["Train", "Validation"], loc="upper right")
    plt.yscale("log")
    save_plot(plt);

plot_training_loss(history_ann_1.history, "SNN (1 layers) model loss")
plot_training_loss(history_ann_2.history, "DNN (2 layers) model loss")
plot_training_loss(history_ann_3.history, "DNN (3 layers) model loss")
plot_training_loss(history_ann_4.history, "DNN (4 layers) model loss")


# Making an inference using the model and the test sample:

# In[97]:


ann_model_1 = load_model("./model_keras/ann_1.h5")
ann_model_2 = load_model("./model_keras/ann_2.h5")
ann_model_3 = load_model("./model_keras/ann_3.h5")
ann_model_4 = load_model("./model_keras/ann_4.h5")


# In[98]:


layers_test["ann_score_1"] = ann_model_1.predict(np.array(test_x))[:, 1]
layers_test["ann_score_2"] = ann_model_2.predict(np.array(test_x))[:, 1]
layers_test["ann_score_3"] = ann_model_3.predict(np.array(test_x))[:, 1]
layers_test["ann_score_4"] = ann_model_4.predict(np.array(test_x))[:, 1]


# In[99]:


# Distribution of scores:
fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, 1, 100)
plt.hist(layers_test[layers_test["score"] < 0]["ann_score_4"], bins=bins, alpha=0.5, label="Normalies")
plt.hist(layers_test[layers_test["score"] > 0]["ann_score_4"], bins=bins, alpha=0.5, label="Anomalies")
plt.title("Distribution of scores: DNN (4 layers)")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
save_plot(plt)


# In[100]:


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
    save_plot(plt);


# In[101]:


cae = autoencoder()
#print("Autoencoder Architecture:")
#cae.summary()


# In[711]:


history_cae = train_nn(cae,
                       train_cae,
                       train_cae,
                       512,
                       keras.losses.mse,
                       "cae",
                       validation_split=0.2)
plot_training_loss(history_cae.history, "Autoencoder model loss")


# In[102]:


cae_model = load_model("./model_keras/cae.h5")
layers_test["cae_score"] = np.sum(abs(test_x - cae_model.predict(np.array(test_x))), axis=1)


# In[103]:


def benchmark(y_true, y_score, treshold):
    y_pred = 2*(y_score > treshold)-1
    y_true = 2*(y_true > treshold)-1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = round(float(tp)/(tp+fn), 4)
    specificity = round(float(tn)/(tn+fp), 4)

    print(("Model accuracy: %s" % round(accuracy_score(y_true, y_pred), 4)))
    print(("Model sensitivity: %s" % sensitivity))
    print(("Model specificity: %s" % specificity))

    return specificity, sensitivity
    
print("DNN:")
specificity_ann, sensitivity_ann = benchmark(layers_test["score"], layers_test["ann_score_4"], 0.5)
print("AE:")
specificity_cae, sensitivity_cae = benchmark(layers_test["score"], layers_test["cae_score"], 0.035)


# In[302]:


get_roc_curve(layers_test, 
              [
               ("AE", "cae_score"),
               ("SNN", "ann_score_1"),
               ("DNN (2 layers)", "ann_score_2"),
               ("DNN (3 layers)", "ann_score_3"),
               ("DNN (4 layers)", "ann_score_4"),
               ("Simple test", "st_score"),
               ], (specificity_ann, sensitivity_ann))


# In[105]:


fig, ax = plt.subplots()
ax.set_yscale('log')
#ax.grid()
bins = np.linspace(0, 0.1, 200)
plt.hist(layers_test[layers_test["score"] < 0]["cae_score"], bins=bins,         alpha=0.5, label="Normal chambers")
plt.hist(layers_test[layers_test["score"] > 0]["cae_score"], bins=bins,         alpha=0.5, label="Anomalous chambers")
plt.title("Distribution of scores: AE")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.03, color='k', linestyle='dashed', linewidth=1)
save_plot(plt)


# Setting thresholds:

# In[153]:


th_dnn = 0.4
th_ae = 0.025


# In[290]:


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
    for i, j in itertools.product(list(range(cm.shape[0])), list(range(cm.shape[1]))):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[155]:


threshold = th_dnn
y_pred = 2*(layers_test["ann_score_4"] > threshold)-1
layers_test["score"] = 2*(layers_test["score"] > threshold)-1

cnf_matrix = confusion_matrix(layers_test["score"], y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix DNN, with normalization')


# In[156]:


threshold = th_ae
y_pred = 2*(layers_test["cae_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix AE, with normalization')


# In[157]:


print("Number of FPs for the DNN:", len(layers_test[(layers_test["score"] == -1) &                                                    (layers_test["ann_score_4"] > th_dnn)]))


# In[158]:


print("Number of FPs for the AE:", len(layers_test[(layers_test["score"] == -1) &                                                   (layers_test["cae_score"] > th_ae)]))


# In[159]:


layers_test["name"] = ("W" + layers_test["wheel"].astype(str) + "_S" + layers_test["sector"].astype(str) +       "_St" + layers_test["station"].astype(str))


# Where are the FPs located?

# In[160]:


def count_fp(df, dis_nn, th, filt):
    df_temp = df[(layers_test["score"] == -1) & (layers_test[dis_nn] > th)].copy()
    df_temp = df_temp.groupby(['run', 'wheel', 'sector', 'station', 'name'])    .size().reset_index(name='counts')
    if filt:
        df_temp = df_temp[df_temp["counts"] > 1]
    #print df_temp
    return len(df_temp), df_temp

num_fp_ann, fp_ann = count_fp(layers_test, "ann_score_4", th_dnn, False)
num_fp_cae, fp_cae = count_fp(layers_test, "cae_score", th_ae, True)

print("Number of chambers with false positives DNN:", num_fp_ann)
print("Number of chambers with false positives AE:", num_fp_cae)
fp_cae.set_index("name",drop=True,inplace=True)
fp_ann.set_index("name",drop=True,inplace=True)


# In[161]:


#fp_ann["counts"].plot(kind='bar', title ="False positives DNN", legend=True, fontsize=12)


# In[162]:


fp_cae["counts"].plot(kind='bar', title ="False positives AE", legend=True, fontsize=12)


# In[382]:


def get_matrix(df):
    x = np.zeros((5,12),dtype=int)
    for i in range(len(df)):
        a = int(5-df["station"].iloc[i])
        b = int(df["sector"].iloc[i]-1)
        x[a,b] = x[a,b] + 1
    return x

def deduceLS(data):
    return data*10+5
layers_test["averageLS"] = layers_test["group"].apply(deduceLS)


# In[380]:


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
    print(mat)
    print(mat.sum())

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
    save_plot(plt)


# In[ ]:


th = th_ae
dis_nn = "cae_score"
rule = (layers_test["score"] == -1) & (layers_test[dis_nn] > th)
plot_scatter(layers_test[rule], 306125, -3, 210, 230)

plot_scatter(layers_test[rule], -1, -2, -1, -1)
plot_scatter(layers_test[rule], -1, -1, -1, -1)
plot_scatter(layers_test[rule], -1, 0, -1, -1)
plot_scatter(layers_test[rule], -1, +1, -1, -1)
plot_scatter(layers_test[rule], -1, +2, -1, -1)


# Where are the FPs located in time?

# In[391]:


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
    save_plot(plt)
    return n


# In[ ]:


threshold = th_ae
n1 = plotFpVsLs(306121, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306121]["ls_end"])
n2 = plotFpVsLs(306122, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306122]["ls_end"])
n3 = plotFpVsLs(306125, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306125]["ls_end"])
n4 = plotFpVsLs(306126, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306126]["ls_end"])


# Trying some benchmark algorithms (for outlier detection). Isolation forest:

# In[166]:


def variance(content):
    return np.var(content)
layers_test["variance_score"] = layers_test["content"].apply(variance)


# In[210]:


def cross_validation_split(train_X, train_y, clf_i, param_grid, return_params=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    clf = GridSearchCV(clf_i, param_grid, cv=skf, scoring='roc_auc'); 
    clf.fit(train_X, train_y)
    if return_params:
        return clf.best_params_
    return clf.best_estimator_


# In[211]:


param_grid = [{"max_samples": [100, 1000],
               "n_estimators": [10, 100],
               "contamination": np.array(list(range(4, 13, 1)))/100.0}]

ifparams = cross_validation_split(np.vstack(layers_train["content_scaled"].values),
                                 -layers_train["score"].astype(int),
                                 IsolationForest(random_state=rng, 
                                                 #verbose=1
                                                ),
                                 param_grid)


# In[212]:


# Retrain IF using all unlabelled samples

ifclf = IsolationForest(max_samples=ifparams.max_samples,
                        n_estimators=ifparams.n_estimators,
                        contamination=ifparams.contamination,
                        random_state=rng)

ifclf.fit(np.vstack(normalies_train["content_scaled"].values))


# Then use SVM for outlier detection:

# In[213]:


# This may take some time... to be run on a remote machine possibly on GPU

#param_grid = [{"nu": np.array(range(1, 10, 1))/10.0,
#               "gamma": ["auto", 0.1, 0.01, 0.001, 0.0001],
#               "kernel": ["linear", "rbf"]}]

#svmparams = cross_validation_split(np.vstack(layers_train["content_scaled"].values),
#                                  -layers_train["score"].astype(int),
#                                  svm.OneClassSVM(random_state=rng, 
#                                                  verbose=1
#                                                 ),
#                                  param_grid)


# In[215]:


# Retrain SVM using only good samples. For the moment using some temporary values.
svmclf = svm.OneClassSVM(
                         #nu=svmparams.nu,
                         nu=0.10000000000000001,
                         #gamma=svmparams.gamma,
                         gamma='auto',
                         kernel='linear',
                         random_state=rng
                        )
svmclf.fit(np.vstack(normalies_train["content_scaled"].values))


# In[216]:


layers_test["svm_score"] = -svmclf.decision_function(np.vstack(layers_test["content_scaled"].values))
layers_test["if_score"] = -ifclf.decision_function(np.vstack(layers_test["content_scaled"].values))


# In[217]:


get_roc_curve(layers_test,[
                           ("Variance", "variance_score"),
                           ("IF", "if_score"),
                           ("SVM", "svm_score"),
                           ("AE", "cae_score"),
                           ("DNN", "ann_score_4"),
                          ]
             )


# In[218]:


from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
bdt_clf = tree.DecisionTreeClassifier()
bdt_clf = bdt_clf.fit(np.vstack(layers_train["content_scaled"].values), layers_train["score"].astype(int))
layers_test["bdt_score"] = bdt_clf.predict(np.vstack(layers_test["content_scaled"].values))


# In[219]:


ada_clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)
ada_clf = ada_clf.fit(np.vstack(layers_train["content_scaled"].values), layers_train["score"].astype(int))
layers_test["ada_score"] = ada_clf.predict(np.vstack(layers_test["content_scaled"].values))


# In[220]:


threshold = 0.0
y_pred = 2*(layers_test["bdt_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix BDT, with normalization')


# In[221]:


threshold = 0.0
y_pred = 2*(layers_test["ada_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix ADA BDT, with normalization')


# Trying KNeighborsClassifier classifier:

# In[222]:


from sklearn import neighbors
knclf = neighbors.KNeighborsClassifier(15, weights='distance')
knclf.fit(np.vstack(layers_train["content_scaled"].values),          -layers_train["score"].astype(int))


# In[223]:


layers_test["knn_score"] = -knclf.predict(np.vstack(layers_test["content_scaled"].values))


# In[224]:


threshold = 0.0
y_pred = 2*(layers_test["knn_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix KNN, with normalization')


# In[226]:


filename = './model_sktlearn/knn.sav'
joblib.dump(knclf, filename)


# Trying unsupervised algorithms: LOF. Optimizing LOF parameters first:

# In[232]:


threshold = 0.0
result_lof = []
for i in range(5, 100, 2):
    lofclf = LocalOutlierFactor(n_neighbors=i, contamination=0.02)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[233]:


for i in range(105, 1000, 100):
    lofclf = LocalOutlierFactor(n_neighbors=i, contamination=0.02)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[234]:


for i in range(1005, 2000, 100):
    lofclf = LocalOutlierFactor(n_neighbors=i, contamination=0.02)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[235]:


fp_rate = []
tp_rate = []
for i in range(len(result_lof)):
    fp_rate.append([result_lof[i][0], result_lof[i][2]*1.0/(result_lof[i][1]+result_lof[i][2])])
    tp_rate.append([result_lof[i][0], result_lof[i][4]*1.0/(result_lof[i][3]+result_lof[i][4])])


# In[239]:


def plot_perf(der, rel, xaxis, yaxis, x1, x2, y1, y2, logx, logy):
    fig, ax = plt.subplots()
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')
    a, b = list(zip(*der))
    c, d = list(zip(*rel))
    plt.ylim(y1, y2)
    plt.xlim(x1, x2)
    plt.plot(a, b, "ro-", alpha=0.5, label = "False positive rate")
    plt.plot(c, d, "mo-", alpha=0.5, label = "True positive rate")
    plt.axhline(1.00, color='k', linestyle='dashed', linewidth=1)
    #plt.axhline(0.01, color='g', linestyle='dashed', linewidth=1)
    if (xaxis == 'Number of neighbors'):
        plt.axvline(1200, color='k', linestyle='dashed', linewidth=1)
    elif (xaxis == 'Contamination'):
        plt.axvline(0.02, color='k', linestyle='dashed', linewidth=1)
    plt.title("LOF")
    plt.legend(loc='best')
    #plt.ylabel('Inertia')
    plt.xlabel(xaxis)
    save_plot(plt)
    
plot_perf(fp_rate, tp_rate, 'Number of neighbors', '', 0, 2000, 0.01, 1.1, False, True)


# In[240]:


threshold = 0.0
result_lof = []
for i in np.array(list(range(1, 500, 5)))/1000.0:
    lofclf = LocalOutlierFactor(n_neighbors=1205, contamination=i)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[241]:


fp_rate_c = []
tp_rate_c = []
for i in range(len(result_lof)):
    fp_rate_c.append([result_lof[i][0], result_lof[i][2]*1.0/(result_lof[i][1]+result_lof[i][2])])
    tp_rate_c.append([result_lof[i][0], result_lof[i][4]*1.0/(result_lof[i][3]+result_lof[i][4])])


# In[242]:


plot_perf(fp_rate_c, tp_rate_c, 'Contamination', '', 0.01, 0.2, 0.001, 1.1, False, True)


# Choosing ...

# In[371]:


# fit the model
lofclf = LocalOutlierFactor(n_neighbors=1205, contamination=0.02)#It has to be a odd number
layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))


# In[373]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(-1, +1, 3)
plt.hist(layers_train[layers_train["score"] < 0]["lof_score"], bins=bins, alpha=0.5, label="Normalies")
plt.hist(layers_train[layers_train["score"] > 0]["lof_score"], bins=bins, alpha=0.5, label="Anomalies")
plt.title("Distribution of scores: LOF")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.0, color='k', linestyle='dashed', linewidth=1)
save_plot(plt)


# In[611]:


threshold = 0.0
y_pred = 2*(layers_train["lof_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_train["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix LOF, with normalization')


# In[612]:


layers_train["averageLS"] = layers_train["group"].apply(deduceLS)
threshold = 0.0
n1 = plotFpVsLs(306121, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306121]["ls_end"])
n2 = plotFpVsLs(306122, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306122]["ls_end"])
n3 = plotFpVsLs(306125, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306125]["ls_end"])
n4 = plotFpVsLs(306126, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306126]["ls_end"])


# Trying KMeans clustering algorithm. First of all let's find the optimal number of clusters (parameter to the algorithm) by scanning the average distance vs. number of clusters:

# In[251]:


means = []
for i in range(1, 300, 10):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content_scaled"].values))
    means.append([i, k_means.inertia_])


# In[252]:


for i in range(300, 500, 10):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content_scaled"].values))
    means.append([i, k_means.inertia_])


# In[253]:


for i in range(500, 1000, 100):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content_scaled"].values))
    means.append([i, k_means.inertia_])


# In[254]:


for i in range(1000, 2000, 200):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content_scaled"].values))
    means.append([i, k_means.inertia_])


# In[255]:


for i in range(2, 10, 1):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content_scaled"].values))
    means.append([i, k_means.inertia_])


# In[256]:


def getKey(item):
    return item[0]

means = sorted(means, key=getKey)


# In[257]:


der = []
for i in range(0, len(means)-1):
    x, y = list(zip(*means))
    a = np.sqrt(y[i]/len(normalies_train))
    b = np.sqrt(y[i+1]/len(normalies_train))
    delta = (a-b)/a
    der.append([x[i], delta])


# In[258]:


rel = []
for i in range(0, len(means)-1):
    x, y = list(zip(*means))
    a = np.sqrt(y[i]/len(normalies_train))
    b = np.sqrt(y[i+1]/len(normalies_train))
    delta = (a-b)*100000/y[0]
    rel.append([x[i], delta])


# In[259]:


def plotDist(means, der, rel):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    x, y = list(zip(*means))
    a, b = list(zip(*der))
    c, d = list(zip(*rel))
    plt.ylim(0.0001, 10.)
    plt.xlim(1, 2000)
    plt.plot(x, np.sqrt(np.array(y)/len(normalies_train)), "bo-", alpha=0.5, label="Average distance")
    plt.plot(a, b, "ro-", alpha=0.5, label = "Relative variation: (d$_{cls}$ - d$_{cls+1}$)/d$_{cls}$")
    plt.plot(c, d, "mo-", alpha=0.5, label = "Relative variation: (d$_{cls}$ - d$_{cls+1}$)/d$_{cls = 1}$    (x$10^{-5}$)")
    plt.axhline(0.05, color='k', linestyle='dashed', linewidth=1)
    plt.axhline(0.01, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(100, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(500, color='y', linestyle='dashed', linewidth=1)
    plt.title("K-Means clustering")
    plt.legend(loc='best')
    #plt.ylabel('Inertia')
    plt.xlabel('Number of clusters')
    save_plot(plt)
    
plotDist(means, der, rel)


# Choosing 450 clusters and training on the normalies only:

# In[724]:


n_cls = 450
k_means = cluster.KMeans(n_clusters=n_cls)

norm = layers_train[layers_train.score == -1].copy()
distances = k_means.fit_transform(np.vstack(norm["content_scaled"].values))
norm["kmeans_score"] = k_means.labels_


# In[725]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, n_cls, n_cls+1)
plt.hist(norm[norm["score"] < 0]["kmeans_score"], bins=bins, alpha=0.5, label="Normalies")
plt.title("K-Means cluster distribution")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Cluster')
save_plot(plt)


# The minimal distance is used to assign the data to the clusters. Visualize the average distance within a cluster:

# In[726]:


minim = []
for i in range(0, len(distances)):
    #print  min(distances[i])
    minim.append(min(distances[i]))


# In[727]:


#print len(minim)
norm["dist"] = minim
#print norm["dist"]


# In[728]:


temp1 = norm.groupby(["kmeans_score"])[["dist"]].mean().reset_index()        


# In[729]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, 0.02, 20)
plt.hist(temp1["dist"], bins=bins, alpha=0.5, label="Normalies")
plt.title("K-Means average distance distribution")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Average distance within clusters')
save_plot(plt)


# Predicting the distances of the test sample data with respect to the clusters and plotting the minimum distance for each point (dividing anomalies and normalies):

# In[730]:


distances_test = k_means.transform(np.vstack(layers_test["content_scaled"].values))
minim = []
for i in range(0, len(distances_test)):
    #print  min(distances_test[i])
    minim.append(min(distances_test[i]))
layers_test["dist"] = minim


# In[731]:


fig, ax = plt.subplots()
ax.set_yscale('log')
#ax.grid()
bins = np.linspace(0, 0.05, 100)
plt.hist(layers_test[layers_test["score"] < 0]["dist"], bins=bins, alpha=0.5, label="Normal chambers")
plt.hist(layers_test[layers_test["score"] > 0]["dist"], bins=bins, alpha=0.5, label="Anomalous chambers")
plt.axvline(0.025, color='k', linestyle='dashed', linewidth=1)
plt.title("K-Means average distance distribution")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Distance to the closest cluster')
save_plot(plt)


# In[732]:


th_km = 0.025
y_pred = 2*(layers_test["dist"] > th_km)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix K-Means, with normalization')


# In[733]:


get_roc_curve(layers_test,[
                           #("Variance", "variance_score"),
                           #("IF", "if_score"),
                           #("SVM", "svm_score"),
                           ("AE", "cae_score"),
                           ("KMeans", "dist"),
                           ("DNN", "ann_score_4"),
                          ]
             )


# In[271]:


filename = './model_sktlearn/kmeans_all.sav'
joblib.dump(k_means, filename)


# ## Reducing features

# Still KMeans but reducing the number of features. Removing lumi, rate and errors: 

# In[315]:


# 0.system, 1.wheel, 2.sector, 3.station, 4.lumi, 5.err, 6.rate, 7.err, 8.CS, 9.err
def removeItem(data):
    index = [0, 4, 5, 6, 7, 9] #leaving only position and CS
    #index = [0, 5, 7, 9] #removing system and errors
    temp = data.copy()
    temp = np.delete(temp, index)
    return temp

layers_test["content"] = layers_test["content"].apply(np.array)
layers_test["content_red"] = layers_test["content"].apply(removeItem)
layers_test["content_red"] = layers_test["content_red"].apply(np.array)

layers_train["content"] = layers_train["content"].apply(np.array)
layers_train["content_red"] = layers_train["content"].apply(removeItem)
layers_train["content_red"] = layers_train["content_red"].apply(np.array)


# Determine the average cross-section for every chamber during a normal run and save the one from station 1:

# In[316]:


values = []
for i in [-2, -1, 0, +1, +2]:
    v_temp = []
    for j in range(1, 13):
        temp = cross_sections[(cross_sections.wheel == i) &                            (cross_sections.sector == j)].copy()
        #print temp
        v_temp.append(temp["CS"].iloc[0])
        #print i, j, temp["CS"].iloc[0]
    values.append(v_temp)


# Calculating scale factors to normalize the CS to 5 in all the chambers:

# In[317]:


factors = layers_train[(layers_train.score == -1) & (layers_train.lumi > 6000)].groupby(['wheel', 'sector', 'station'])["CS"].mean().reset_index()
factors["factor"] = factors["CS"].copy()

def scale_cs_for_sf(df):
    for i in [-2, -1, 0, +1, +2]:
        for j in range(1, 13):
            for k in range(1, 6):
                rule = (df.wheel == i) & (df.sector == j) & (df.station == k)
                indexes = df[rule].index
                if (len(indexes) == 0):
                    continue            
                for m in indexes:
                  df.loc[m, "factor"] = 5/df.loc[m, "CS"]


# In[318]:


scale_cs_for_sf(factors)


# In[319]:


print(factors[(factors.wheel == +1) & (factors.sector == 4) & (factors.station == 3)])
print(factors[(factors.wheel == +2) & (factors.sector == 10) & (factors.station == 3)])


# In[320]:


factors.to_csv("factors.csv", sep=',', header=False)


# Multiplying the cross-section values by the factors previously determined:

# In[321]:


#Reindex because otherwise anomalies and normalies have the same labels/indexes in W+3_S4_MB3
layers_train.index = pd.RangeIndex(len(layers_train.index))
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

scale_cs(layers_train, factors)
scale_cs(layers_test, factors)


# In[322]:


#0.No rescaling
#1.RobustScaler
#2.MaxAbsScaler 
#3.QuantileTransformerUniform: NO
#4.QuantileTransformerNormal: NO
#5.Normalizer
#6.MinMaxScaler

scaler_type = 5
layers_test["content_red_scaled"] = layers_test["content_red"].apply(scale_data, args=[scaler_type])
layers_train["content_red_scaled"] = layers_train["content_red"].apply(scale_data, args=[scaler_type])


# Extracting CS from the feature vector and plotting the scaled CS:

# In[402]:


def extract_cs(data):
    #print data[0][3]
    #return data[3]
    return data[0][3]

#layers_train["CS_red"] = layers_train["content_red"].apply(extract_cs)
#layers_test["CS_red"] = layers_test["content_red"].apply(extract_cs)

layers_train["CS_red"] = layers_train["content_red_scaled"].apply(extract_cs)
layers_test["CS_red"] = layers_test["content_red_scaled"].apply(extract_cs)


# In[403]:


plot_scatter_2(layers_test[layers_test.run == 306125], "CS_red", -2)
plot_scatter_2(layers_test[layers_test.run == 306125], "CS_red", -1)
plot_scatter_2(layers_test[layers_test.run == 306125], "CS_red", 0)
plot_scatter_2(layers_test[layers_test.run == 306125], "CS_red", +1)
plot_scatter_2(layers_test[layers_test.run == 306125], "CS_red", +2)


# In[325]:


print(layers_train["content_red"].iloc[1])
print(layers_train["content_red_scaled"].iloc[1])
print(layers_test["content_red"].iloc[1])
print(layers_test["content_red_scaled"].iloc[1])


# Lumi oscillations in run 306125:

# In[280]:


r1 = [200, 255]
r2 = [2265, 2315]


# In[326]:


norm = layers_train[layers_train.score == -1].copy()
#norm = norm[norm.run == 306125]


# In[348]:


n_cls = 450
k_means_red = cluster.KMeans(n_clusters=n_cls)
distances_red = k_means_red.fit_transform(np.vstack(norm["content_red"].values))
norm["kmeans_red_score"] = k_means_red.labels_


# In[349]:


minim = []
for i in range(0, len(distances_red)):
    #print  min(distances_red[i])
    minim.append(min(distances_red[i]))


# In[350]:


#print len(minim)
norm["dist_red"] = minim
#print norm["dist_red"]


# In[351]:


temp1 = norm.groupby(["kmeans_red_score"])[["dist_red"]].mean().reset_index()


# In[352]:


distances_test_red = k_means_red.transform(np.vstack(layers_test["content_red"].values))
minim = []
for i in range(0, len(distances_test_red)):
    #print  min(distances_test_red[i])
    minim.append(min(distances_test_red[i]))
layers_test["dist_red"] = minim


# In[376]:


fig, ax = plt.subplots()
ax.set_yscale('log')
#ax.grid()
bins = np.linspace(0, 2.0, 200)
#rule = (layers_test.wheel == +1) & (layers_test.sector == 3) & (layers_test.station == 3)
plt.hist(layers_test[layers_test["score"] < 0]["dist_red"], bins=bins,         alpha=0.5, label="Normal chambers")
plt.hist(layers_test[layers_test["score"] > 0]["dist_red"], bins=bins,         alpha=0.5, label="Anomalous chambers")
#plt.hist(layers_test[(layers_test["score"] < 0) & rule]["dist_red"], bins=bins,\
#         alpha=0.5, label="Normal chambers")
plt.axvline(0.4, color='k', linestyle='dashed', linewidth=1)
plt.title("K-Means average distance distribution")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Distance to the closest cluster')
save_plot(plt)


# In[333]:


th_km = 0.4
y_pred = 2*(layers_test["dist_red"] > th_km)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix K-Means, with normalization')


# In[356]:


get_roc_curve(layers_test,[
                           #("Variance", "variance_score"),
                           #("IF", "if_score"),
                           #("SVM", "svm_score"),
                           ("AE", "cae_score"),
                           ("KMeans", "dist_red"),
                           ("DNN", "ann_score_4"),
                           ("Simple test", "st_score"),
                          ]
             )


# In[334]:


filename = './model_sktlearn/kmeans_red.sav'
joblib.dump(k_means_red, filename)


# In[389]:


dis_nn = "dist_red"
th_km = 0.4
rule = (layers_test["score"] == -1) & (layers_test[dis_nn] > th_km)

plot_scatter(layers_test[rule], -1, -2, 0, -1)
plot_scatter(layers_test[rule], -1, -1, 0, -1)
plot_scatter(layers_test[rule], -1, 0, 0, -1)
plot_scatter(layers_test[rule], -1, +1, 0, -1)
plot_scatter(layers_test[rule], -1, +2, 0, -1)


# In[394]:


layers_test["averageLS"] = layers_test["group"].apply(deduceLS)
threshold = 0.4
plotFpVsLs(306125, 0, 0, 0, "Distribution of false positives: K-Means clustering, ",           layers_test, "dist_red",           threshold, True, boundaries[boundaries["run"] == 306125]["ls_end"])
plotFpVsLs(306126, 0, 0, 0, "Distribution of false positives: K-Means clustering, ",           layers_test, "dist_red",           threshold, True, boundaries[boundaries["run"] == 306126]["ls_end"])


# ## LOF reduced

# In[363]:


# fit the model
lofclf = LocalOutlierFactor(n_neighbors=201, contamination=0.02)#It has to be a odd number
layers_test["lof_score_red"] = -lofclf.fit_predict(np.vstack(layers_test["content_red"].values))


# In[377]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(-1, +1, 3)
plt.hist(layers_test[layers_test["score"] < 0]["lof_score_red"], bins=bins, alpha=0.5, label="Normalies")
plt.hist(layers_test[layers_test["score"] > 0]["lof_score_red"], bins=bins, alpha=0.5, label="Anomalies")
plt.title("Distribution of scores: LOF")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.axvline(0.0, color='k', linestyle='dashed', linewidth=1)
save_plot(plt)


# In[ ]:




