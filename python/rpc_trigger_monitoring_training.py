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
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy import ndimage, misc, stats
import datetime
from datetime import timedelta

import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Reshape, Conv1D, MaxPooling1D, AveragePooling1D,UpSampling1D, InputLayer

from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.externals import joblib
from sklearn.neighbors import LocalOutlierFactor
from sklearn import cluster, datasets


# In[2]:


from pandas.tseries import converter
converter.register() 


# In[3]:


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

# In[4]:


#import os
#cwd = os.getcwd()
#print(cwd)


# In[5]:


lumi_directory = "./lumi"
rates_directory = "./rates"
runs = [297101, 297178, 297179, 297180, 297181]
#runs = [297181]
#runs_ref = [297179]
runs_ref = [297219, 299481, 300122, 300155, 300157, 300576, 300636, 300785, 301959, 301987, 301998, 302163,            302263, 302448, 302573, 302597, 302635, 303832, 303838, 303885, 303948, 304062, 304125, 304144,            304158, 304169, 304292, 304333, 304366, 304447, 304508, 304655, 304671, 304738, 304778, 304797,            305064, 305081, 305112, 305188, 305204, 305207, 305237, 305366, 305377, 305406, 305518, 305590,            305636, 305814, 305840, 306125, 306135, 306138, 306139, 306154, 306155, 306459]


# # Reading cvs files <a class="anchor" id="first-bullet"></a>

# Reading instantaneous luminosities from the cvs file produced with brilcalc and saving into a pandas dataframe:

# In[6]:


df_rates = pd.DataFrame()
int_lumi2 = pd.DataFrame()
for run in runs:
    print(("Loading %s" % run))
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

print(df_rates[df_rates["run"]==1])

# In[7]:


for run in runs_ref:
    print(("Loading %s" % run))
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


# In[8]:


df_certified = pd.DataFrame()
print("Loading info")
path = "runs_2017.csv"
df_certified = df_certified.append(pd.read_csv(path,names=["runfill", "time", "nls", "ncms", "delivered", "recorded"]), ignore_index=True)
print("Done.")


# In[9]:


df_certified = df_certified.drop(index = [0, 1, 473, 474, 475, 476, 477])
df_certified['run'], df_certified['fill'] = df_certified['runfill'].str.split(':', 1).str


# In[10]:
df_certified.reset_index(drop=True, inplace=True)
print(df_certified["runfill"])

#print(df_certified.ncms[1])
print(df_certified[df_certified["ncms"] >1000])
nLS = 1000
print(("Number of runs with more than %s LS': %i" % (nLS, len(df_certified[df_certified.ncms > nLS].ncms.values))))
df_certified.ncms = df_certified.ncms.astype('int')
df_certified.run = df_certified.run.astype('int')

fig, ax = plt.subplots()
ax.set_yscale('log')
bins = np.linspace(0, 3000, 100)
plt.hist(df_certified.ncms, bins=bins, alpha=0.5, label="Number of LS'")
plt.axvline(nLS, color='k', linestyle='dashed', linewidth=1)
plt.ylabel('Number of runs')
plt.xlabel('Number of LS')
plt.legend(loc='best')
save_plot(plt)

print("Runs with more than 1000 LS':", df_certified[df_certified.ncms > nLS].run.values)


# # Luminosity section <a class="anchor" id="second-bullet"></a>

# Dropping useless rows inherited from the lumi CVS file:

# In[11]:


int_lumi2["source"] = int_lumi2["source"].astype('str')
int_lumi2 = int_lumi2[int_lumi2["source"] != "nan"]
int_lumi2 = int_lumi2[int_lumi2["source"] != "source"]


# Splitting run:fill field and the start and end lumi sections:

# In[12]:


int_lumi2['run'], int_lumi2['fill'] = int_lumi2['runfill'].str.split(':', 1).str
int_lumi2['ls_start'], int_lumi2['ls_end'] = int_lumi2['ls'].str.split(':', 1).str


# Converting run to integer and luminosities to float:

# In[13]:


int_lumi2["run"] = int_lumi2["run"].astype('int')
int_lumi2["ls_start"] = int_lumi2["ls_start"].astype('int')
int_lumi2["ls_end"] = int_lumi2["ls_end"].astype('int')
int_lumi2["delivered"] = int_lumi2["delivered"].astype('float64')
int_lumi2["recorded"] = int_lumi2["recorded"].astype('float64') 


# Converting time stamp to datetime:

# In[14]:


def transform_time(data):
    from datetime import datetime
    time_str = data.time
    #print time_str
    datetime_object = datetime.strptime(time_str, "%m/%d/%y %H:%M:%S")
    #print datetime_object
    return datetime_object
int_lumi2["time"] = int_lumi2.apply(transform_time, axis=1);


# Creating end time column from the start time:

# In[15]:


int_lumi2["time_end"] = int_lumi2["time"]


# Finding the runs and their start and end times:

# In[16]:


boundaries = pd.DataFrame(columns=["run", "start", "end", "ls_start", "ls_end", "nLS"])
for i in runs+runs_ref:
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


# In[17]:


boundaries = boundaries.sort_values('run')
boundaries = boundaries.reset_index()


# Reindexing the dataframe after removing some lines:

# In[18]:


int_lumi2.index = pd.RangeIndex(len(int_lumi2.index))


# In[19]:


print(len(int_lumi2.index))


# Filling end time column:

# In[20]:


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

# In[21]:


#print int_lumi2["beamstatus"]
int_lumi2 = int_lumi2[int_lumi2["beamstatus"] == "STABLE BEAMS"]


# In[22]:


#int_lumi2.to_csv("int_lumi2.csv", sep='\t')


# Plotting the instantaneous luminosities:

# In[23]:


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


# In[24]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 297179]["time"], 
               int_lumi2[int_lumi2["run"] == 297179]["delivered"], 
               int_lumi2[int_lumi2["run"] == 297179]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               ("297181", int_lumi2["fill"].iloc[0])))


# In[25]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 297180]["time"], 
               int_lumi2[int_lumi2["run"] == 297180]["delivered"], 
               int_lumi2[int_lumi2["run"] == 297180]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               ("297180", int_lumi2["fill"].iloc[0])))


# In[26]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 297181]["time"], 
               int_lumi2[int_lumi2["run"] == 297181]["delivered"], 
               int_lumi2[int_lumi2["run"] == 297181]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               ("297181", int_lumi2["fill"].iloc[0])))


# In[27]:


plot_inst_lumi(int_lumi2[int_lumi2["run"] == 306125]["time"], 
               int_lumi2[int_lumi2["run"] == 306125]["delivered"], 
               int_lumi2[int_lumi2["run"] == 306125]["recorded"], 
               ("Instantaneous Luminosity for Run / Fill: %s / %s" % 
               ("306125", int_lumi2[int_lumi2.run == 306125]["fill"].iloc[0])))


# # Trigger rate section <a class="anchor" id="third-bullet"></a>

# Converting columns to proper data types:

# In[28]:


df_rates["time"] = pd.to_datetime(df_rates["time"])
df_rates["run"] = df_rates["run"].astype('int')
#print df_rates["time"]


# Splitting, converting and adding new columns:

# In[ ]:


df_rates['wheel'], df_rates['sector'] = df_rates['board'].str.split('_', 1).str
df_rates["wheel"] = df_rates["wheel"].astype(str)
df_rates["sector"] = df_rates["sector"].astype(str)


# In[ ]:


df_rates["wheel"].replace(regex=True,inplace=True,to_replace=r'YB',value=r'')
df_rates["sector"].replace(regex=True,inplace=True,to_replace=r'S',value=r'')
df_rates["wheel"] = df_rates["wheel"].astype('int')
df_rates["sector"] = df_rates["sector"].astype('int')
df_rates["ls"] = -1
df_rates["lumi"] = -1.0
df_rates["score"] = -1
#df_rates.to_csv("df_rates.csv", sep='\t')


# Plotting the rate coming from one of the stations:

# In[ ]:


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
    save_plot(plt)


# In[ ]:


rule = (df_rates.run == 306125)
boards = ["YB-1_S4", "YB-1_S5", "YB-1_S6"]
for board in boards:
    plot_rate_vs_time(df_rates[rule],                      "time", "RPC1", board, "Rates for Runs / Fill / Board: %s / %s / %s" % 
                      ("306125", int_lumi2[int_lumi2.run == 306125]["fill"].iloc[0], board))


# Associating a LS and an instantaneous luminosity to each rate:

# In[ ]:


#Just a backup copy
df_rates_backup = df_rates.copy()


# In[ ]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# Removing the measurements taken before and after the start and end time reported by the brilcalc output. All the 60 boards are measured at the same time. In order to speed-up the association, we take just one board, the first one. This reduces the dataframe and the time needed to go though it by a factor of 60.

# In[ ]:

#Workaround to use the correct boundaries for each run
# In[74]:
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


# In[ ]:


print(len(df_rates_noduplicates))


# Assigning the LS and the inst. lumi. to the measurements for the selected board:

# In[ ]:


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

# In[138]:


df_rates_noduplicates = df_rates_noduplicates[df_rates_noduplicates["ls"] > 0]
print(len(df_rates_noduplicates))


# Save in a csv file:

# In[139]:


#df_rates.to_csv("df_rates.csv", sep='\t')
#df_rates_noduplicates.to_csv("df_rates_nodup.csv", sep='\t')


# In[140]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# Assign the LS and the inst. lumi. to all the 60 boards for each time:

# In[141]:


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


# In[142]:


df_rates = temp.copy()
#print df_rates[df_rates.ls <= 0]


# Removing measurements without LS assignment:

# In[143]:


df_rates_backup = df_rates.copy()
df_rates = df_rates[df_rates.ls > 0]
#print df_rates["ls"]


# In[144]:


#print df_rates[df_rates.ls <= 0]
#df_rates.to_csv("df_rates.csv", sep='\t')


# Averaging the rates associated to the same LS:

# In[145]:


df_boards = df_rates.copy()
df_boards = df_boards.groupby(['board']).size().reset_index(name='counts')
print(len(df_boards))
#print df_boards


# Too slow to use all the measurements. Averaging over 10 LS:

# In[146]:


#Uncomment to restore backup copy
#df_rates = df_rates_backup.copy()


# In[147]:


bunch = 50
def assignGroup(data, div = bunch):
    res = int(data/div)
    #print data, res
    return res

df_rates["group"] = df_rates["ls"]
df_rates["group"] = df_rates["group"].apply(assignGroup)


# In[148]:


#print df_rates["group"]


# In[149]:


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

# In[150]:


import math
def applySqrt(data):
    return math.sqrt(data)

df_rates["counts"] = df_rates["counts"].apply(applySqrt)

for i in list(df_rates):
    if "err" in i:
        #print i
        df_rates[i] = df_rates[i]/df_rates["counts"]


# In[151]:


#print df_rates


# Check for null or NaN values:

# In[152]:


print(df_rates.isnull().values.any())
null_columns=df_rates.columns[df_rates.isnull().any()]
print((df_rates[df_rates.isnull().any(axis=1)][null_columns].head()))
#df_rates = df_rates.fillna(0)
#print(df_rates[df_rates.isnull().any(axis=1)][null_columns].head())


# In[153]:


#Another backup
#df_rates_backup = df_rates.copy()
#df_rates.to_csv("df_rates.csv", sep='\t')


# In[154]:


#Restore backup
#df_rates = df_rates_backup.copy()


# In[155]:


print(len(df_rates))


# Uncomment to check just one case case:

# In[156]:


#for index, row in df_rates.iterrows():
    #if row["board"] == "YB0_S1":
        #print "Index:", index,", Run:", row["run"],", Board: ",row["board"],",\
        #LS: ",row["ls"],", Rate: ",row["DT1"],", Error: ",row["errDT1"]


# Plotting the result:

# In[157]:


def calc_bounds(x, a, b, s_a, s_b, mode):
    if (mode == "low"):
        return (x * (a - s_a) + (b - s_b))
    else:
        return (x * (a + s_a) + (b + s_b))

def plot_rate_vs_ls(df, run, x_val, y_val, z_val, x_err, y_err, title_x, title_y, title, opt, log,                    fit = True, fmin = 5000, fmax = 15000):
    df_temp = df.copy()
    fig, ax = plt.subplots()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    if log:
        ax.set_yscale('log')
    inter = []
    for i in range(len(run)):
        rule = ((df_temp["board"] == z_val) & (df_temp["run"] == run[i]))
        newstr = opt[i].replace("o", "")
        plt.errorbar(df_temp[rule][x_val], df_temp[rule][y_val], xerr=x_err,        yerr=df_temp[rule][y_err], fmt=opt[i], ecolor=newstr, label=str(run[i]))
        plt.legend(loc="best")
        
        if ((run[i] == 306125) & fit):
            rule2 = (df_temp[x_val] > fmin) & (df_temp[x_val] < fmax)
            val1 = df_temp[rule & rule2][x_val]
            #print val1
            val2 = df_temp[rule & rule2][y_val]
            print("Linear correlation:", stats.pearsonr(val1.values, val2.values))
            coeff, pcov = np.polyfit(val1, val2, 1, cov = True)
            #print pcov
            perr3 = np.sqrt(np.diag(pcov))
            inter.append(coeff)
            inter.append(perr3)
            p = np.poly1d(coeff)
            xp = np.linspace(3000, 16000, 1000)
            _ = plt.plot(xp, p(xp), '-')
            print("Linear fit parameters:", coeff)
            print("Paramenters uncertainty:", perr3)
            y_low = calc_bounds(val1, coeff[0], coeff[1], perr3[0], perr3[1], "low")
            y_high = calc_bounds(val1, coeff[0], coeff[1], perr3[0], perr3[1], "high")
            plt.plot(val1, y_low, 'r-')
            plt.plot(val1, y_high, 'r-')
        
    plt.title(title)
    save_plot(plt)
    if ((run[i] == 306125) & fit):
        return inter


# In[158]:


#boards = ["YB-1_S4", "YB-1_S5", "YB-1_S6"]    
    
title = "Rates for Fill/Run/Board: "+str(int_lumi2[int_lumi2.run == 306125]["fill"].iloc[0])+" / "+str(boundaries["run"].iloc[3])+" / YB-1_S4"

plot_rate_vs_ls(df_rates, [306125, 297180], "lumi", "RPC1", "YB-1_S4", 0, "errRPC1",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, ["ro", "bo"], False)
plot_rate_vs_ls(df_rates, [306125], "lumi", "RPC2", "YB-1_S4", 0, "errRPC2",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, ["ro"], False)
plot_rate_vs_ls(df_rates, [306125], "lumi", "RPC3", "YB-1_S4", 0, "errRPC3",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, ["ro"], False)
plot_rate_vs_ls(df_rates, [306125], "lumi", "RPC4", "YB-1_S4", 0, "errRPC4",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                "Rate [Hz]", title, ["ro"], False)


# Create a new dataframe with the input features already organized in a numpy array:

# In[159]:


print(len(df_rates))
algos = ['RPC1', 'RPC2', 'RPC3', 'RPC4']
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

# In[160]:


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


# In[161]:


array = df_rates_new_2.as_matrix(columns=['system', 'wheel', 'sector', 'station',                                        'lumi', 'errLumi', 'rate', 'err',                                        'CS', 'errCS'])


# In[162]:


df_rates_new_2["content"] = np.empty((len(df_rates_new_2), 0)).tolist()
for index, rows in df_rates_new_2.iterrows():
    #print index, array[index]
    df_rates_new_2.at[index, "content"] = array[index]
df_rates_new_2["score"] = -1


# Check if the two dataframes are exactly the same:

# In[163]:


def plot_rate_vs_ls_2(df1, df2, x_val, y_val, x_err, y_err, title_x, title_y, title, opt, log = False):
    fig, ax = plt.subplots()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    if log:
        ax.set_yscale('log')
    plt.errorbar(df1[x_val], df1[y_val], xerr=x_err, yerr=df1[y_err], fmt='ro', ecolor='r')
    num = y_val
    num = num.replace("RPC", "")
    tmp = df2[df2.station == int(num)]
    plt.errorbar(tmp[x_val], tmp["rate"], xerr=x_err, yerr=tmp["err"], fmt='b+', ecolor='b')
    plt.title(title)
    plt.legend(loc="best")
    save_plot(plt)


# In[164]:


title = "Fill/Run/Board: "+str(int_lumi2[int_lumi2.run == 306125]["fill"].iloc[0])+" / "+str(boundaries["run"].iloc[3])+" / YB-1_S4"

rule_1 = ((df_rates["wheel"] == -1) & (df_rates["sector"] == 4) & (df_rates["run"] == 306125))
rule_2 = ((df_rates_new_2["wheel"] == -1) & (df_rates_new_2["sector"] == 4)          & (df_rates_new_2["run"] == 306125))

temp1 = df_rates[rule_1]
temp2 = df_rates_new_2[rule_2]

plot_rate_vs_ls_2(temp1, temp2, "lumi", "RPC1", 0, "errRPC1", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "lumi", "RPC2", 0, "errRPC2", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "lumi", "RPC3", 0, "errRPC3", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "lumi", "RPC4", 0, "errRPC4", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Rate [Hz]", title, "ro")


# In[165]:


plot_rate_vs_ls_2(temp1, temp2, "ls", "RPC1", 0, "errRPC1", "LS", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "ls", "RPC2", 0, "errRPC2", "LS", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "ls", "RPC3", 0, "errRPC3", "LS", 
                "Rate [Hz]", title, "ro")
plot_rate_vs_ls_2(temp1, temp2, "ls", "RPC4", 0, "errRPC4", "LS", 
                "Rate [Hz]", title, "ro")


# Checking cross-sections for some chambers:

# In[166]:


title = "Fill/Run/Board: "+str(int_lumi2[int_lumi2.run == 306125]["fill"].iloc[0])+" / "+str(boundaries["run"].iloc[3])+" / YB-1_S4"

plot_rate_vs_ls(df_rates_new_2, [306125], "lumi", "CS", "YB-1_S6", 0, "errCS",                r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
                "Cross-section [cm$^{2}$]", title, ["ro"], False, False)


# Defining the fit ranges for all the chambers:

# In[167]:


df_rates_new_2["fmin"] = 5000
df_rates_new_2["fmax"] = 15000
df_rates_new_2["slope"] = 9999
df_rates_new_2["inter"] = 9999
df_rates_new_2["err_a"] = 9999
df_rates_new_2["err_b"] = 9999
df_rates_new_2["rate_low"] = df_rates_new_2["rate"].copy()
df_rates_new_2["rate_high"] = df_rates_new_2["rate"].copy()
df_rates_new_2["rate_mid"] = df_rates_new_2["rate"].copy()


# In[168]:


def set_range(df, wh, sec, st, fmin, fmax):
    rule = (df_rates_new_2.wheel == wh) &           (df_rates_new_2.sector == sec) &           (df_rates_new_2.station == st)
    indexes = df[rule].index
    for m in indexes:
        df.loc[m, "fmin"] = fmin
        df.loc[m, "fmax"] = fmax


# In[169]:


set_range(df_rates_new_2, -2, 7, 3, 4000, 10000)
set_range(df_rates_new_2, -2, 10, 3, 8000, 15000)
set_range(df_rates_new_2, 0, 3, 2, 5000, 12000)
set_range(df_rates_new_2, 0, 9, 2, 5000, 10000)
set_range(df_rates_new_2, +1, 8, 1, 6000, 15000)
set_range(df_rates_new_2, +1, 8, 3, 5000, 15000)


# In[170]:


def plot_scatter_2(df, arg, wheel, norm = False, show = True, fit = False, fmin = 5000, fmax = 15000):
    wheel_s = "All"
    temp = df.copy()
    if wheel != -3:
        temp = temp[(temp["wheel"] == wheel)]
        wheel_s = str(wheel)
    else:
        return 1
    
    temp = temp.groupby(['run', 'wheel', 'sector', 'system', 'station'])    [[arg]].mean().reset_index()
    #print temp
    
    if fit:
        for j in range(1, 13):
            for k in range(1, 5):
    
                rule = (df.wheel == wheel) & (df.sector == j) & (df.station == k) & (df.system == 1)
                rule2 = (df.lumi > df.fmin) & (df.lumi < df.fmax)
                val1 = df[rule & rule2]["lumi"]
                val2 = df[rule & rule2]["rate"]
                coeff = np.polyfit(val1, val2, 1)
                #print "Linear fit parameters:", wheel, j, k, coeff[0]

                rule = (temp.sector == j) & (temp.station == k) & (temp.system == 1)
                ind = temp[rule].index
                temp.loc[ind[0], arg] = coeff[0]
                #print temp.loc[ind[0]]
    
    mat = []
    for i in [4, 3, 2, 1]:
        rule = (temp.station == i)
        temp2 = temp[rule].sort_values(["sector"], ascending=True)
        vec = list(temp2[arg].values)
        mat.append(vec)      
        #print i, temp[rule][arg].values
    if norm:
        mat = mat/np.matrix(mat).sum()
        mat = [100*i for i in mat]
    #print mat

    plt.figure()
    
    ax = plt.gca()
    ax.set_yticklabels(["1", "2", "3", "4"])
    ax.set_yticks([3, 2, 1, 0])
    ax.set_xticklabels(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    
    plt.xlabel("Sector")
    plt.ylabel("Station")
    title = "Wheel: "+wheel_s
    plt.title(title, loc="left")   

    im = ax.imshow(mat, interpolation="nearest")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    if show:
        for i in range(0,4):
            for j in range(0,12):
                if (mat[i][j] == 0):
                    ax.text(j, i, -1,                    ha="center", va="center", color="r")
                else:
                    text = ax.text(j, i, round(mat[i][j], 2),                    ha="center", va="center", color="w")
    
    plt.colorbar(im, cax=cax, ticks=[np.min(np.nan_to_num(mat)), np.max(np.nan_to_num(mat))])
    save_plot(plt)


# Checking the slope of the rate vs. inst. lumi. for each chamber during the good reference run:

# In[171]:


#plot_scatter_2(df_rates_new_2[df_rates_new_2.run == 306125], "CS", -2, False, True, True)
#plot_scatter_2(df_rates_new_2[df_rates_new_2.run == 306125], "CS", -1, False, True, True)
#plot_scatter_2(df_rates_new_2[df_rates_new_2.run == 306125], "CS", 0, False, True, True)
#plot_scatter_2(df_rates_new_2[df_rates_new_2.run == 306125], "CS", +1, False, True, True)
#plot_scatter_2(df_rates_new_2[df_rates_new_2.run == 306125], "CS", +2, False, True, True)


# Plotting the cross-sections for the good and anomalous runs on the same canvas:

# In[172]:


def plot_ratio_vs_ls(df, run, x_val, y_val, z_val, x_err, y_err, title_x, title_y, title, opt, log,                     fit = False, fmin = 5000, fmax = 15000):
    df_temp = df.copy()
    fig, ax = plt.subplots()
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    ax.grid()
    if log:
        ax.set_yscale('log')
    df_temp["ratio"] = df_temp[y_val]/df_temp[x_val]
    df_temp["errRatio"] = (1/df_temp[x_val])*    np.sqrt(df_temp[y_err]**2 + df_temp["errLumi"]**2*df_temp["ratio"]**2)
    inter = []
    for i in range(len(run)):
        rule = ((df_temp["board"] == z_val) & (df_temp["run"] == run[i]))
        if ((run[i] == 306125) & fit):
            rule2 = (df_temp[x_val] > fmin) & (df_temp[x_val] < fmax)
            val1 = df_temp[rule & rule2][x_val]
            val2 = df_temp[rule & rule2]["ratio"]
            coeff = np.polyfit(val1, val2, 2)
            inter.append(coeff)
            p = np.poly1d(coeff)
            xp = np.linspace(3000, 16000, 1000)
            _ = plt.plot(xp, p(xp), '-')
        
        newstr = opt[i].replace("o", "")
        plt.errorbar(df_temp[rule][x_val], df_temp[rule]["ratio"], xerr=x_err,        yerr=df_temp[rule]["errRatio"], fmt=opt[i], ecolor=newstr, label=str(run[i]))
        
    print("Linear fit for the reference run:", inter[0])
    plt.legend(loc="best")
    plt.title(title)
    save_plot(plt)


# In[173]:


#for i in [-2]: #Only anomalous wheels
    #for j in range(7, 8): #Only anomalous sectors
        #range_ = range(3, 4)
for i in [-2, -1, 0, +1, +2]:
    for j in range(1, 13):
        range_ = list(range(1, 5))
        for k in range_:

            if (i > 0):
                board = "YB+" + str(i) + "_S" + str(j)
            else:
                board = "YB" + str(i) + "_S" + str(j)
            station = "RPC" + str(k)
            
            error = "err"+station
            
            title = board + "_RB" + str(k)
                
            print(title)
            
            fmin = df_rates_new_2[(df_rates_new_2.station == k) & (df_rates_new_2.board == board)]["fmin"].iloc[0]
            fmax = df_rates_new_2[(df_rates_new_2.station == k) & (df_rates_new_2.board == board)]["fmax"].iloc[0]
            #print fmin, fmax
                
            #plot_ratio_vs_ls(df_rates_new_2[df_rates_new_2.station == k],\
            #                [297179, 297180, 297181, 306125],\
            #                 "lumi", "rate", board, 0,\
            #                "err", r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]", 
            #                r"Cross-section [$\times10^{-30}$ cm$^{2}$]",\
            #                 title, ["mo", "go", "ro", "bo"], False, True, fmin, fmax)

            coeff = plot_rate_vs_ls(df_rates_new_2[df_rates_new_2.station == k],                                    [306459, 297101, 297178, 297179, 297180, 297181, 306125],                                    "lumi", "rate", board, 0, "err",                                    r"Inst. Lumi. [$\times10^{30}$ Hz/cm$^2$]",                                    "Rate [Hz]", title, ["ko", "co", "yo", "mo", "go", "ro", "bo"],                                    False, True, fmin, fmax)
            
            #print coeff
            rule = (df_rates_new_2.station == k) & (df_rates_new_2.board == board) & (df_rates_new_2.run == 306125)
            indexes = df_rates_new_2[rule].index
            for m in indexes:
                a = coeff[0][0]
                b = coeff[0][1]
                s_a = coeff[1][0]
                s_b = coeff[1][1]
                df_rates_new_2.loc[m, "slope"] = a
                df_rates_new_2.loc[m, "inter"] = b
                df_rates_new_2.loc[m, "err_a"] = s_a
                df_rates_new_2.loc[m, "err_b"] = s_b
                x = df_rates_new_2.loc[m, "lumi"]
                df_rates_new_2.loc[m, "rate_mid"] = x*a + b #central value
                df_rates_new_2.loc[m, "rate_low"] = x*(a - s_a) + (b - s_b) #lower limit
                df_rates_new_2.loc[m, "rate_high"] = x*(a + s_a) + (b + s_b) #upper limit
            print("----------------------------------------------------------------------------------------------------------------")


# In[174]:


def get_matrix(df):
    x = np.zeros((5,12),dtype=int)
    for i in range(len(df)):
        a = int(5-df["station"].iloc[i])
        b = int(df["sector"].iloc[i]-1)
        x[a,b] = x[a,b] + 1
    return x


# In[175]:


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


# Calculating the number of hits as the area under the curve Rate vs. LS. Each point is the average rate over 10 LS.

# In[176]:


def calcHits(data):
    # 230 is 10 lumisections times 23.3 seconds, the duration of one LS
    return data*10*23.3

df_rates_new_2["hits"] = df_rates_new_2["rate"].apply(calcHits)
#print df_rates_new_2["hits"]


# In[177]:


temp = df_rates_new_2.groupby(["wheel", "sector", "station", "run", "system"])[["hits"]].sum().reset_index()        


# Normalizing the number of hits for each chamber to the total number of hits collected during the run:. N.B.: the numbers are percentages.

# In[178]:


#plot_scatter_2(temp[temp.run == 306125], "hits", -2, True)
#plot_scatter_2(temp[temp.run == 306125], "hits", -1, True)
#plot_scatter_2(temp[temp.run == 306125], "hits", 0, True)
#plot_scatter_2(temp[temp.run == 306125], "hits", +1, True)
#plot_scatter_2(temp[temp.run == 306125], "hits", +2, True)


# In[179]:


#plot_scatter_2(temp[temp.run == 297179], "hits", -2, True)
#plot_scatter_2(temp[temp.run == 297179], "hits", -1, True)
#plot_scatter_2(temp[temp.run == 297179], "hits", 0, True)
#plot_scatter_2(temp[temp.run == 297179], "hits", +1, True)
#plot_scatter_2(temp[temp.run == 297179], "hits", +2, True)


# # Model training section <a class="anchor" id="fourth-bullet"></a>

# In[180]:


print(df_rates_new_2.columns)


# ## Creating train and test samples:

# In[219]:


anomalies = df_rates_new_2.copy()


# In[220]:


normalies = anomalies.copy()


# In[221]:


print(len(normalies), len(anomalies))


# In[222]:


def assignScore(df, wheel, sector, station, run, ls_min, ls_max, score):
    temp = df.copy()
    
    rule = (temp["run"] > 0)
    
    if(run != -1):
        rule = rule & (temp["run"] == run)
        
    if(wheel != -1):
        rule = rule & (temp["wheel"] == wheel)
    if(sector != -1):
        rule = rule & (temp["sector"] == sector)
    if(station != -1):
        rule = rule & (temp["station"] == station)
    
    if ((ls_min != -1) & (ls_max == -1)):
        rule = rule & (temp["ls"] >= ls_min)
    elif ((ls_min == -1) & (ls_max != -1)):
        rule = rule & (temp["ls"] <= ls_max)
    elif ((ls_min != -1) & (ls_max != -1)):
        rule = rule & (temp["ls"] >= ls_min) & (temp["ls"] <= ls_max)
        
    #print rule
    indexes = temp[rule].index
    #print indexes
    for i in indexes:
        temp.loc[i, "score"] = score
    return temp


# During runs 297179, 297180, 297181 LV tripped in: W-1_S4, all stations; W-1_S5, all stations; W-1_S6, all stations.

# In[223]:


temp = assignScore(anomalies, -1, 4, -1, 297179, -1, -1, 1)
anomalies = temp
temp = assignScore(anomalies, -1, 5, -1, 297179, -1, -1, 1)
anomalies = temp
temp = assignScore(anomalies, -1, 6, -1, 297179, -1, -1, 1)
anomalies = temp


# In[224]:


temp = assignScore(anomalies, -1, 4, -1, 297180, -1, -1, 1)
anomalies = temp
temp = assignScore(anomalies, -1, 5, -1, 297180, -1, -1, 1)
anomalies = temp
temp = assignScore(anomalies, -1, 6, -1, 297180, -1, -1, 1)
anomalies = temp


# In[225]:


temp = assignScore(anomalies, -1, 4, -1, 297181, -1, -1, 1)
anomalies = temp
temp = assignScore(anomalies, -1, 5, -1, 297181, -1, -1, 1)
anomalies = temp
temp = assignScore(anomalies, -1, 6, -1, 297181, -1, -1, 1)
anomalies = temp


# Check that the change affects only normalies:

# In[226]:


rule = (normalies["wheel"] == -1) & (normalies["sector"] == 3) &(normalies["station"] == 3) & (normalies["run"] == 297180)
print("Normal chamber:")
print(normalies[rule]["score"].iloc[0])

rule = (normalies["wheel"] == -1) & (normalies["sector"] == 4) &(normalies["station"] == 3) & (normalies["run"] == 297180)
print("Anomalous chamber:")
print(normalies[rule]["score"].iloc[0])


# In[227]:


rule = (anomalies["wheel"] == -1) & (anomalies["sector"] == 3) &(anomalies["station"] == 3) & (normalies["run"] == 297180)
print("Normal chamber:")
print(anomalies[rule]["score"].iloc[0])

rule = (anomalies["wheel"] == -1) & (anomalies["sector"] == 4) &(anomalies["station"] == 3) & (normalies["run"] == 297180)
print("Anomalous chamber:")
print(anomalies[rule]["score"].iloc[0])


# In[228]:


rule = (anomalies["wheel"] == -1) & (anomalies["sector"] == 3) &(anomalies["station"] == 3) & (normalies["run"] == 306125)
print("Normal chamber:")
print(anomalies[rule]["score"].iloc[0])

rule = (anomalies["wheel"] == -1) & (anomalies["sector"] == 4) &(anomalies["station"] == 3) & (normalies["run"] == 306125)
print("Anomalous chamber:")
print(anomalies[rule]["score"].iloc[0])


# In[229]:


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


# In[230]:


anomalies["content"] = anomalies["content"].apply(np.array)
anomalies["content_scaled"] = anomalies["content"].apply(scale_data)

normalies["content"] = normalies["content"].apply(np.array)
normalies["content_scaled"] = normalies["content"].apply(scale_data)


# In[231]:


#print anomalies["content_scaled"]
#print normalies["content_scaled"]
normalies = normalies[normalies.run != 297179]
normalies = normalies[normalies.run != 297180]
normalies = normalies[normalies.run != 297181]


# In[235]:


# Set a random seed to reproduce the results
rng = np.random.RandomState(0)
anomalies = anomalies[(anomalies.score == 1)]
normalies = normalies[(normalies.score == -1)]
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


# In[236]:


print(("Number of anomalies in the train set: %s" % len(anomalies_train)))
print(("Number of normal in the train set: %s" % len(normalies_train)))
print(("Number of anomalies in the test set: %s" % len(anomalies_test)))
print(("Number of normal in the test set: %s" % len(normalies_test)))


# In[237]:


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


# In[293]:


def cae_generate_input():
    return (np.array(np.concatenate(normalies.content_scaled.values)).reshape(-1, 10),
            np.array(np.concatenate(normalies[normalies.run == 306125].content_scaled.values)).reshape(-1, 10))

train_cae, train_cae_ref = cae_generate_input()


# In[239]:


from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils import class_weight

cw = class_weight.compute_class_weight("balanced",
                                       np.unique(np.argmax(train_y, axis=1)),
                                       np.argmax(train_y, axis=1))
cw = {0: cw[0], 1: cw[1]}
print(cw)


# ## Simple test

# In[255]:


# Distribution of scores:
def plot_distr_score(df, model, title, th, log = True):
    fig, ax = plt.subplots()
    if log:
        ax.set_yscale('log')
    ax.grid()
    bins = np.linspace(0, 1.0, 100)
    plt.hist(df[df["score"] < 0][model], bins=bins, alpha=0.5,         label="Normal chambers")
    plt.hist(df[df["score"] > 0][model], bins=bins, alpha=0.5,         label="Anomalous chambers")
    plt.title(title)
    plt.legend(loc='best')
    plt.ylabel('Frequency')
    plt.xlabel('Score')
    plt.axvline(th, color='k', linestyle='dashed', linewidth=1)
    save_plot(plt)


# In[275]:


def simple_test(df_info, df_test, coeff = 0):
    for i in df_test.index:
        wheel = df_test.wheel.loc[i]
        sector = df_test.sector.loc[i]
        station = df_test.station.loc[i]
        lumi = df_test.lumi.loc[i]
        rate = df_test.rate.loc[i]

        rule = ((df_info.wheel == wheel) & (df_info.sector == sector) &                (df_info.station == station))
        
        a = (df_info[rule].slope).iloc[0]
        b = (df_info[rule].inter).iloc[0]
        s_a = (df_info[rule].err_a).iloc[0]
        s_b = (df_info[rule].err_b).iloc[0]
        
        mid = lumi*a + b #central value
        down = lumi*(a - s_a) + (b - s_b) #lower limit
        up = lumi*(a + s_a) + (b + s_b) #upper limit
        
        #print rate, down, mid, up

        delta = abs(rate - mid) / mid
        err1 = abs(up - mid) / mid
        err2 = abs(down - mid) / mid
        err = (err1 + err2) / 2
        
        #print delta, err
        
        res = 0
        if ((rate > (1 - coeff)*down) & (rate < (1 + coeff)*up)):
            res = delta
        else:
            res = 1
            #print rate, down, mid, up, err, delta
        df_test.loc[i, "st_score"] = res


# In[280]:


layers_test_2 = layers_test.copy()
layers_test_2 = layers_test_2[layers_test_2.run == 306125]

layers_test_2["st_score"] = 0
layers_test_2.index = pd.RangeIndex(len(layers_test_2.index))
simple_test(df_info, layers_test_2, 0.5)


# In[274]:


plot_distr_score(layers_test_2, "st_score", "Distribution of scores: Simple Test", 1)


# In[278]:


df_info = normalies.copy()
df_info = df_info[df_info.run == 306125] #The assumption is that the fits are performed using 306125 only

layers_test["st_score"] = 0
layers_test.index = pd.RangeIndex(len(layers_test.index))
simple_test(df_info, layers_test, 0.5)


# In[279]:


plot_distr_score(layers_test, "st_score", "Distribution of scores: Simple Test", 1)


# In[328]:


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


# ## NN architectures:

# In[281]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())
K.set_session(sess)


# Training the NN:

# In[309]:


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


# In[310]:


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
    save_plot(plt)


# In[311]:


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


# In[315]:


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


# In[330]:


def autoencoder():
    from keras.layers import Input, Dense
    from keras.models import Model
    input_ = Input(shape=(10,))
    encoded = Dense(10, activation='relu')(input_)
    encoded = Dense(9, activation='relu')(encoded)
    encoded = Dense(8, activation='relu')(encoded)
    encoded = Dense(7, activation='relu')(encoded)
    encoded = Dense(6, activation='relu')(encoded)
    encoded = Dense(5, activation='relu')(encoded)
    encoded = Dense(4, activation='relu')(encoded)

    decoded = Dense(4, activation='relu')(encoded)
    decoded = Dense(5, activation='relu')(decoded)
    decoded = Dense(6, activation='relu')(decoded)
    decoded = Dense(7, activation='relu')(decoded)
    decoded = Dense(8, activation='relu')(decoded)
    decoded = Dense(9, activation='relu')(decoded)
    decoded = Dense(10, activation='sigmoid')(decoded)

    autoencoder = Model(input_, decoded)
    return autoencoder


# Making an inference using the model and the test sample:

# In[331]:


cae = autoencoder()
#print("Autoencoder Architecture:")
#cae.summary()


# In[332]:


history_cae = train_nn(cae,
                       train_cae_ref,
                       train_cae_ref,
                       512,
                       keras.losses.mse,
                       "cae",
                       validation_split=0.2)
plot_training_loss(history_cae.history, "Autoencoder model loss")


# In[314]:


cae_model = load_model("./model_keras/cae.h5")
layers_test["cae_score"] = np.sum(abs(test_x - cae_model.predict(np.array(test_x))), axis=1)


# In[316]:


print("AE:")
specificity_cae, sensitivity_cae = benchmark(layers_test["score"], layers_test["cae_score"], 0.035)


# In[318]:


plot_distr_score(layers_test, "cae_score", "Distribution of scores: Autoencoder", 0.2)


# In[327]:


get_roc_curve(layers_test, 
              [
               ("Autoencoder", "cae_score"),
               ("Simple test", "st_score"),
               ], (specificity_cae, sensitivity_cae))


# In[329]:


threshold = th_ae = 0.085
y_pred = 2*(layers_test["cae_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix AE, with normalization')


# Studying the false positives

# In[ ]:


print("Number of FPs for the AE:", len(layers_test[(layers_test["score"] == -1) &                                                   (layers_test["cae_score"] > th_ae)]))


# In[ ]:


layers_test["name"] = ("W" + layers_test["wheel"].astype(str) + "_S" + layers_test["sector"].astype(str) +       "_St" + layers_test["station"].astype(str))


# Where are the FPs located?

# In[ ]:


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


# In[ ]:


#fp_ann["counts"].plot(kind='bar', title ="False positives DNN", legend=True, fontsize=12)


# In[ ]:


fp_cae["counts"].plot(kind='bar', title ="False positives AE", legend=True, fontsize=12)


# In[ ]:


def deduceLS(data):
    return data*10+5
layers_test["averageLS"] = layers_test["group"].apply(deduceLS)


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

# In[ ]:


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
    
threshold = th_ae
n1 = plotFpVsLs(306121, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306121]["ls_end"])
n2 = plotFpVsLs(306122, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306122]["ls_end"])
n3 = plotFpVsLs(306125, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306125]["ls_end"])
n4 = plotFpVsLs(306126, 0, 0, 0, "Distribution of false positives: AE, ", layers_test, "cae_score",                threshold, True, boundaries[boundaries["run"] == 306126]["ls_end"])


# Trying some benchmark algorithms (for outlier detection). Isolation forest first:

# In[151]:


def cross_validation_split(train_X, train_y, clf_i, param_grid, return_params=False):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rng)
    clf = GridSearchCV(clf_i, param_grid, cv=skf, scoring='roc_auc'); 
    clf.fit(train_X, train_y)
    if return_params:
        return clf.best_params_
    return clf.best_estimator_


# In[152]:


param_grid = [{"max_samples": [100, 1000],
               "n_estimators": [10, 100],
               "contamination": np.array(list(range(4, 13, 1)))/100.0}]

ifparams = cross_validation_split(np.vstack(layers_train["content_scaled"].values),
                                 -layers_train["score"].astype(int),
                                 IsolationForest(random_state=rng, 
                                                 #verbose=1
                                                ),
                                 param_grid)


# In[153]:


# Retrain IF using all unlabelled samples

ifclf = IsolationForest(max_samples=ifparams.max_samples,
                        n_estimators=ifparams.n_estimators,
                        contamination=ifparams.contamination,
                        random_state=rng)

ifclf.fit(np.vstack(normalies_train["content_scaled"].values))


# Then use SVM for outlier detection:

# In[154]:


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


# In[155]:


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


# In[156]:


layers_test["svm_score"] = -svmclf.decision_function(np.vstack(layers_test["content_scaled"].values))
layers_test["if_score"] = -ifclf.decision_function(np.vstack(layers_test["content_scaled"].values))


# In[157]:


get_roc_curve(layers_test,[
                           ("IF", "if_score"),
                           ("SVM", "svm_score"),
                           ("AE", "cae_score"),
                          ]
             )


# Trying unsupervised algorithms: LOF. Optimizing LOF parameters first:

# In[ ]:


threshold = 0.0
result_lof = []
for i in range(5, 100, 2):
    lofclf = LocalOutlierFactor(n_neighbors=i, contamination=0.02)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[ ]:


for i in range(105, 1000, 100):
    lofclf = LocalOutlierFactor(n_neighbors=i, contamination=0.02)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[ ]:


for i in range(1005, 2000, 100):
    lofclf = LocalOutlierFactor(n_neighbors=i, contamination=0.02)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[ ]:


fp_rate = []
tp_rate = []
for i in range(len(result_lof)):
    fp_rate.append([result_lof[i][0], result_lof[i][2]*1.0/(result_lof[i][1]+result_lof[i][2])])
    tp_rate.append([result_lof[i][0], result_lof[i][4]*1.0/(result_lof[i][3]+result_lof[i][4])])


# In[ ]:


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


# In[ ]:


threshold = 0.0
result_lof = []
for i in np.array(list(range(1, 500, 5)))/1000.0:
    lofclf = LocalOutlierFactor(n_neighbors=1205, contamination=i)#It has to be a odd number
    layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))
    y_pred = 2*(layers_train["lof_score"] > threshold)-1
    tn, fp, fn, tp = confusion_matrix(layers_train["score"].astype(int), y_pred).ravel()
    result_lof.append([i, tn, fp, fn, tp])


# In[ ]:


fp_rate_c = []
tp_rate_c = []
for i in range(len(result_lof)):
    fp_rate_c.append([result_lof[i][0], result_lof[i][2]*1.0/(result_lof[i][1]+result_lof[i][2])])
    tp_rate_c.append([result_lof[i][0], result_lof[i][4]*1.0/(result_lof[i][3]+result_lof[i][4])])


# In[ ]:


plot_perf(fp_rate_c, tp_rate_c, 'Contamination', '', 0.01, 0.2, 0.001, 1.1, False, True)


# Choosing ...

# In[160]:


# fit the model
lofclf = LocalOutlierFactor(n_neighbors=1205, contamination=0.02)#It has to be a odd number
layers_train["lof_score"] = -lofclf.fit_predict(np.vstack(layers_train["content_scaled"].values))


# In[161]:


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
save_plot(plt)


# In[162]:


threshold = 0.0
y_pred = 2*(layers_train["lof_score"] > threshold)-1
cnf_matrix = confusion_matrix(layers_train["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix LOF, with normalization')


# In[ ]:


layers_train["averageLS"] = layers_train["group"].apply(deduceLS)
threshold = 0.0
plotFpVsLs(306121, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306121]["ls_end"])
plotFpVsLs(306122, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306122]["ls_end"])
plotFpVsLs(306125, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306125]["ls_end"])
plotFpVsLs(306126, 0, 0, 0, "Distribution of false positives: LOF, ", layers_train, "lof_score",           threshold, True, boundaries[boundaries["run"] == 306126]["ls_end"])


# Trying KMeans clustering algorithm. First of all let's find the optimal number of clusters (parameter to the algorithm) by scanning the average distance vs. number of clusters:

# In[402]:


means = []
for i in range(1, 300, 50):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content"].values))
    means.append([i, k_means.inertia_])


# In[404]:


for i in range(300, 500, 100):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content"].values))
    means.append([i, k_means.inertia_])


# In[405]:


for i in range(500, 1000, 200):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content"].values))
    means.append([i, k_means.inertia_])


# In[406]:


for i in range(1000, 2000, 500):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content"].values))
    means.append([i, k_means.inertia_])


# In[407]:


for i in range(2, 10, 2):
    print(i)
    k_means = cluster.KMeans(n_clusters=i)
    k_means.fit(np.vstack(normalies_train["content"].values))
    means.append([i, k_means.inertia_])


# In[408]:


def getKey(item):
    return item[0]

means = sorted(means, key=getKey)


# In[409]:


der = []
for i in range(0, len(means)-1):
    x, y = list(zip(*means))
    a = np.sqrt(y[i]/len(normalies_train))
    b = np.sqrt(y[i+1]/len(normalies_train))
    delta = (a-b)/a
    der.append([x[i], delta])


# In[410]:


rel = []
for i in range(0, len(means)-1):
    x, y = list(zip(*means))
    a = np.sqrt(y[i]/len(normalies_train))
    b = np.sqrt(y[i+1]/len(normalies_train))
    delta = (a-b)*100000/y[0]
    rel.append([x[i], delta])


# In[418]:


def plotDist(means, der, rel):
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    x, y = list(zip(*means))
    a, b = list(zip(*der))
    c, d = list(zip(*rel))
    plt.ylim(0.000000001, 200000.)
    plt.xlim(1, 2000)
    plt.plot(x, np.sqrt(np.array(y)/len(normalies_train)), "bo-", alpha=0.5, label="Average distance")
    plt.plot(a, b, "ro-", alpha=0.5, label = "Relative variation: (d$_{cls}$ - d$_{cls+1}$)/d$_{cls}$")
    plt.plot(c, d, "mo-", alpha=0.5, label = "Relative variation: (d$_{cls}$ - d$_{cls+1}$)/d$_{cls = 1}$    (x$10^{-5}$)")
    plt.axhline(0.05, color='k', linestyle='dashed', linewidth=1)
    plt.axhline(0.1, color='g', linestyle='dashed', linewidth=1)
    plt.axvline(100, color='k', linestyle='dashed', linewidth=1)
    plt.axvline(240, color='y', linestyle='dashed', linewidth=1)
    plt.title("K-Means clustering")
    plt.legend(loc='best')
    #plt.ylabel('Inertia')
    plt.xlabel('Number of clusters')
    save_plot(plt)
    
plotDist(means, der, rel)


# Choosing 450 clusters and training on the normalies only:

# In[390]:


n_cls = 450
k_means = cluster.KMeans(n_clusters=n_cls)
#distances = k_means.fit_transform(np.vstack(normalies["content_scaled"].values))
distances = k_means.fit_transform(np.vstack(normalies["content"].values))
normalies["kmeans_score"] = k_means.labels_


# In[391]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, n_cls, n_cls+1)
plt.hist(normalies[normalies["score"] < 0]["kmeans_score"], bins=bins, alpha=0.5, label="Normalies")
plt.title("K-Means cluster distribution")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Cluster')
save_plot(plt)


# The minimal distance is used to assign the data to the clusters. Visualize the average distance within a cluster:

# In[392]:


minim = []
for i in range(0, len(distances)):
    #print  min(distances[i])
    minim.append(min(distances[i]))


# In[393]:


#print len(minim)
normalies["dist"] = minim
#print normalies["dist"]


# In[394]:


temp1 = normalies.groupby(["kmeans_score"])[["dist"]].mean().reset_index()        


# In[396]:


fig, ax = plt.subplots()
ax.set_yscale('log')
ax.grid()
bins = np.linspace(0, 10000, 100)
plt.hist(temp1["dist"], bins=bins, alpha=0.5, label="Normalies")
plt.title("K-Means average distance distribution")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Average distance within clusters')
save_plot(plt)


# Predicting the distances of the test sample data with respect to the clusters and plotting the minimum distance for each point (dividing anomalies and normalies):

# In[397]:


#distances_test = k_means.transform(np.vstack(layers_test["content_scaled"].values))
distances_test = k_means.transform(np.vstack(layers_test["content"].values))

minim = []
for i in range(0, len(distances_test)):
    #print  min(distances_test[i])
    minim.append(min(distances_test[i]))
layers_test["dist"] = minim


# In[398]:


fig, ax = plt.subplots()
ax.set_yscale('log')
#ax.grid()
bins = np.linspace(0, 10000, 100)
plt.hist(layers_test[layers_test["score"] < 0]["dist"], bins=bins, alpha=0.5, label="Normal chambers")
plt.hist(layers_test[layers_test["score"] > 0]["dist"], bins=bins, alpha=0.5, label="Anomalous chambers")
plt.axvline(0.025, color='k', linestyle='dashed', linewidth=1)
plt.title("K-Means average distance distribution")
plt.legend(loc='best')
plt.ylabel('Frequency')
plt.xlabel('Distance to the closest cluster')
save_plot(plt)


# In[401]:


th_km = 2000
y_pred = 2*(layers_test["dist"] > th_km)-1
cnf_matrix = confusion_matrix(layers_test["score"].astype(int), y_pred)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["normaly","anomaly"], normalize=True,
                      title='Confusion matrix K-Means, with normalization')


# In[399]:


get_roc_curve(layers_test,[
                           #("IF", "if_score"),
                           #("SVM", "svm_score"),
                           ("AE", "cae_score"),
                           ("KMeans", "dist"),
                          ]
             )


# In[ ]:


filename = './model_sktlearn/kmeans.sav'
joblib.dump(k_means, filename)


# In[ ]:




