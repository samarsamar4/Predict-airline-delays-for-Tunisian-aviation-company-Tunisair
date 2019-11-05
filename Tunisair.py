# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:51:10 2019

@author: Administrator
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
#import statsmoldes as stat
import seaborn as sns 
from sklearn.preprocessing import Imputer 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime, time
dtst1 = pd.read_csv("Train.csv")

x=dtst1.drop('target',axis=1)
y=dtst1['target']
#nbre des pays et distance entre eux 
pays_DEP=x['DEPSTN'].drop_duplicates()
pays_AR=x['ARRSTN'].drop_duplicates()
#distance=
#Durée de flight
#classement selon Operation Date
def season_of_date(date):
    year = str(date.year)
    seasons = {'spring': pd.date_range(start='21/03/'+year, end='20/06/'+year),
               'summer': pd.date_range(start='21/06/'+year, end='22/09/'+year),
               'autumn': pd.date_range(start='23/09/'+year, end='20/12/'+year)}
    if date in seasons['spring']:
        return 'spring'
    if date in seasons['summer']:
        return 'summer'
    if date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'
seasons = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 1]
month_to_season = dict(zip(range(1,13), seasons))
date_lst=[]
season_lst=[]
duree_lst=[]
i=0 
for i in  range(len(x)-1):
      date_lst.append(datetime.datetime.strptime(x['DATOP'][i], '%Y-%m-%d'))
       duree_lst=duree_lst.append(time.mktime(datetime.datetime.strptime(x['STA'][i], '%Y-%m-%d %H.%M.%S').timetuple())-time.mktime(datetime.datetime.strptime(x['STD'][i], '%Y-%m-%d %H:%M:%S').timetuple()))         
      season_lst[i]=(season_of_date(date_lst[i]))
     
data=x.drop('STA', axis=1)
data=data.drop('STD', axis=1)
data=data.drop('DATOP', axis=1)
data=data.drop('ID', axis=1)
data=data.drop( 'FLTID', axis=1)
data.append(season_lst,duree_lst)

# Assuming df has a date column of type `datetime`

  
    
#Duree=x['STA']-x['STD']
#labelisation des données
labenc=LabelEncoder()
x=x.apply(labenc.fit_transform)
#Standarisation des données
scaler=StandardScaler()
scaler.fit(x)

#ville=set(x['DEPSTN'])
#split DaTA
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.6,test_size=0.4)
#The model
#ploter la correlation entre features and labels 
"""plt.figure()
for i, col in enumerate(x.columns):
    plt.subplot(1,9,i+1)
    x=dtst1[col]
    y=y
    plt.plot(x,y,'o')  

#ploter la ligne de regression 
plt.plot(np.unique(x), np.plot(np.polyfit(x,y,1))(np.unique(x)))
plt.style.use('bmh')"""

