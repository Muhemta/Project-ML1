# -*- coding: utf-8 -*-
"""
Created on Sun May 11 01:12:46 2025

@author: Emin Taş
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_curve

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Heart.csv")
df_head = df.head()
df_describe = df.describe()
df_info = df.info()

#missing value analysis
missingvalue = df.isnull().sum()


#Unique Value Analysis
df.columns
for i in list(df.columns):
    print("{} --{}".format(i,df[i].value_counts().shape[0]))
    

#Categorical Feature Analysis
categorical_list = ["sex", "cp","fbs","restecg","exang","slope","ca","thal","target"]

df_categoric = df.loc[:, categorical_list]
for i in categorical_list:
    plt.figure()
    sns.countplot(x = i, data = df_categoric, hue = "target")
    plt.title(i)    