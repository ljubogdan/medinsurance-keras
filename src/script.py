DATA_PATH = 'data/insurance.csv'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#!pip install jupyterthemes # for kaggle notebook (or jupyter)

from jupyterthemes import jtplot

jtplot.style(theme = 'monokai', context = 'notebook', ticks = True, grid = False) 

insurance_df = pd.read_csv(DATA_PATH)

insurance_df
#insurance_df.info()

# exploratory data analysis

insurance_df.isnull() # any null elements?? -no (everything is false)
insurance_df.isnull().sum() # no missing elements

#sns.heatmap(insurance_df.isnull(), yticklabels=False, cbar=False, cmap="Blues") # any null values on heatmap? -no

#insurance_df.describe()

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

# perform data visualization

#insurance_df[['age', 'sex', 'bmi', 'children', 'smoker', 'charges']].hist(bins=30, figsize=(12,12))
#sns.pairplot(insurance_df)

#plt.figure(figsize = (15, 6))
#sns.regplot(x='age', y='charges', data=insurance_df)
#plt.show()

#plt.figure(figsize = (15, 6))
#sns.regplot(x='bmi', y='charges', data=insurance_df)
#plt.show()

# PRACTICE: Calculate and plot correlation matrix. 

insurance_df.corr()

plt.figure(figsize = (10,10))
#sns.heatmap(insurance_df.corr())

# training and testing datasets

#insurance_df.columns

X = insurance_df.drop(columns=['charges'])
y = insurance_df['charges']

X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

y = y.reshape(-1, 1)

print(X.shape)
y.shape

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# PRACTICE: split data 20% testing 80% training (DANGER ZONE: POTENTIONAL DATA LEAKAGE)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

print(X_train.shape)

# train and evaluate linear regression model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

regression_model_sklearn = LinearRegression()
regression_model_sklearn.fit(X_train, y_train)

rmsa = regression_model_sklearn.score(X_test, y_test)
print(rmsa) # accuracy, could be better (about 69%), because we use a lot of unecessary informations

y_predict = regression_model_sklearn.predict(X_test)

y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

k = X_test.shape[1]
n = len(X_test)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

ANN_model = keras.Sequential()
ANN_model.add(Dense(50, input_dim = 8))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dropout(0.5))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(50))
ANN_model.add(Dropout(0.5))
ANN_model.add(Activation('linear'))
ANN_model.add(Dense(1))
ANN_model.compile(loss = 'mse', optimizer = 'adam')
ANN_model.summary()

epochs_hist = ANN_model.fit(X_train, y_train, epochs = 100, batch_size = 20, validation_split = 0.2)

result = ANN_model.evaluate(X_test, y_test)
accuracy_ANN = 1-result
print(accuracy_ANN)

plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])

y_predict = ANN_model.predict(X_test)

"""
plt.plot(y_test, y_predict, "^", color = 'r')
plt.xlabel('Model Predictions')
plt.ylabel('True Values')
"""

# rmse, mse, mae, r2, adj_r2 retry ----> pretty much same result, around 69%

y_predict_orig = scaler_y.inverse_transform(y_predict)
y_test_orig = scaler_y.inverse_transform(y_test)

k = X_test.shape[1]
n = len(X_test)

RMSE = float(format(np.sqrt(mean_squared_error(y_test_orig, y_predict_orig)),'.3f'))
MSE = mean_squared_error(y_test_orig, y_predict_orig)
MAE = mean_absolute_error(y_test_orig, y_predict_orig)
r2 = r2_score(y_test_orig, y_predict_orig)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)
 
# save the model
ANN_model.save('insurance_ann_model.h5')
