DATA_PATH = 'data/insurance.csv'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from jupyterthemes import jtplot

jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False) 

insurance_df = pd.read_csv(DATA_PATH)

insurance_df
insurance_df.info()

# exploratory data analysis

insurance_df.isnull() # any null elements?? -no (everything is false)
insurance_df.isnull().sum() # no missing elements

sns.heatmap(insurance_df.isnull(), yticklabels=False, cbar=False, cmap="Blues") # any null values on heatmap? -no

insurance_df.describe()

df_region = insurance_df.groupby(by = 'region').mean(numeric_only = True) # cannot calculate mean of string values...
df_region

# PRACTICE: Group data by age and examine relationships between age and charges

df_age = insurance_df.groupby(by = 'age').mean(numeric_only = True)
df_age

# perform feature engineering

insurance_df['sex'].unique()

insurance_df['sex'] = insurance_df['sex'].apply(lambda x: 0 if x == "female" else 1)
insurance_df['smoker'] = insurance_df['smoker'].apply(lambda x: 0 if x == "no" else 1) 

insurance_df.head()

region_dummies = pd.get_dummies(insurance_df['region'], drop_first = True).astype(int) # eliminate excess of data if everything is 0
region_dummies

insurance_df = pd.concat([insurance_df, region_dummies], axis=1) # axis=1 means concatenate columns next to eachother (0 means rows)
insurance_df.drop(['region'], axis=1, inplace=True)
insurance_df










