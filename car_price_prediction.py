# -*- coding: utf-8 -*-
"""car_price_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jGjde2tEeouxGvB-4JWJZ71sy968hAjW

Data Understanding
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
pd.set_option("display.max_rows", None,"display.max_columns", None)
warnings.simplefilter(action='ignore')
plt.style.use('seaborn')

#load dataset
df_main = pd.read_csv('/content/CAR DETAILS FROM CAR DEKHO.csv.xlsx - CAR DETAILS FROM CAR DEKHO.csv.xls-2.csv')

df_main.shape

df_main.info()

#numerical stats
df_main.describe()

#missing values
df_main.isna().sum()

"""Data Preprocessing"""

df_main['age'] = 2024 - df_main['year']
df_main.drop('year',axis=1,inplace = True)

df_main.rename(columns = {'owner':'past_owners'},inplace = True)

df_main.head()

df_main.columns

cat_cols = ['fuel','seller_type','transmission','past_owners']
i=0
while i < 4:
    fig = plt.figure(figsize=[10,4])
    #ax1 = fig.add_subplot(121)
    #ax2 = fig.add_subplot(122)

    #ax1.title.set_text(cat_cols[i])
    plt.subplot(1,2,1)
    sns.countplot(x=cat_cols[i], data=df_main)
    i += 1

    #ax2.title.set_text(cat_cols[i])
    plt.subplot(1,2,2)
    sns.countplot(x=cat_cols[i], data=df_main)
    i += 1

    plt.show()

df_main[df_main['km_driven'] > df_main['km_driven'].quantile(0.99)]

df_main[df_main['selling_price'] > df_main['selling_price'].quantile(0.99)]

"""Data Preparation"""

df_main.drop(labels='name',axis= 1, inplace = True)

df_main.head()

df_main['fuel'].unique()

dic_fuel = {'Petrol' : 1, "Diesel" : 2, 'CNG' : 3, 'LPG' : 4, 'Electric' : 5}

df_main['fuel'] = [dic_fuel[i] for i in df_main['fuel']]

df_main['seller_type'].unique()

dic_seller_type = {'Individual' : 1, 'Dealer' : 2, 'Trustmark Dealer' : 3}

df_main['seller_type'] = [dic_seller_type[i] for i in df_main['seller_type']]

df_main['transmission'].unique()

dic_transmission = {'Manual' : 1, 'Automatic' :2}

df_main['transmission'] = [dic_transmission[i] for i in df_main['transmission']]

df_main.head()

#df_main = pd.get_dummies(data = df_main,drop_first=True)

df_main.head()

"""Train-Test Split"""

# Separating target variable and its features
y = df_main['selling_price']
X = df_main.drop('selling_price',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("x train: ",X_train.shape)
print("x test: ",X_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

"""Modeling"""

model = LinearRegression()
model.fit(X_train,y_train)

y_predict = model.predict(X_test)

y_predict

error_list = [actual_data - pre_data for actual_data, pre_data in zip(y_test, y_predict)]
absolute_error_list=[ abs(i) for i in error_list]
print(absolute_error_list)

mean_absolute_error = sum(absolute_error_list)/len(error_list)
print(mean_absolute_error)

error_square_list=[i**2 for i in error_list]
mean_square_error=sum(error_square_list)/len(error_square_list)
print(mean_square_error)

CV = []
R2_train = []
R2_test = []
MAE_train = []
MAE_test = []

def car_pred_model(model, model_name):
    # Training model
    model.fit(X_train, y_train)

    # R2 score of train set
    y_pred_train = model.predict(X_train)
    R2_train_model = r2_score(y_train, y_pred_train)
    R2_train.append(round(R2_train_model, 2))

    # MAE of train set
    #MAE_train_model = mean_absolute_error(y_train, y_pred_train)
    #MAE_train.append(round(MAE_train_model, 2))

    # R2 score of test set
    y_pred_test = model.predict(X_test)
    R2_test_model = r2_score(y_test, y_pred_test)
    R2_test.append(round(R2_test_model, 2))

    # MAE of test set
    #MAE_test_model = mean_absolute_error(y_test, y_pred_test)
    #MAE_test.append(round(MAE_test_model, 2))

    # R2 mean of train set using Cross validation
    cross_val = cross_val_score(model, X_train, y_train, cv=5)
    cv_mean = cross_val.mean()
    CV.append(round(cv_mean, 2))

    # Printing results
    print("Train R2-score:", round(R2_train_model, 2))
    #print("Train MAE:", round(MAE_train_model, 2))
    print("Test R2-score:", round(R2_test_model, 2))
    #print("Test MAE:", round(MAE_test_model, 2))
    print("Train CV scores:", cross_val)
    print("Train CV mean:", round(cv_mean, 2))

    # Plotting Graphs
    # Residual Plot of train data
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].set_title('Residual Plot of Train samples')
    sns.distplot((y_train - y_pred_train), hist=False, ax=ax[0])
    ax[0].set_xlabel('y_train - y_pred_train')

    # Y_test vs Y_train scatter plot
    ax[1].set_title('y_test vs y_pred_test')
    ax[1].scatter(x=y_test, y=y_pred_test)
    ax[1].set_xlabel('y_test')
    ax[1].set_ylabel('y_pred_test')

    plt.show()

lr = LinearRegression()
car_pred_model(lr,"Linear_regressor.pkl")

# Creating Ridge model object
rg = Ridge()
# range of alpha
alpha = np.logspace(-3,3,num=14)

# Creating RandomizedSearchCV to find the best estimator of hyperparameter
rg_rs = RandomizedSearchCV(estimator = rg, param_distributions = dict(alpha=alpha))

car_pred_model(rg_rs,"ridge.pkl")

Technique = ["LinearRegression","Ridge"]
results=pd.DataFrame({'Model': Technique,'R Squared(Train)': R2_train,'R Squared(Test)': R2_test,'CV score mean(Train)': CV})
display(results)

import pickle

# Save the trained model to a pickle file in Google Colab
#model = '/content/carpriceprediction.pkl'
#with open(model, 'wb') as file:
    #pickle.dump(model, file)

#print(f"Model saved to {model}")

import pickle
with open(r"/content/carpriceprediction.pkl", "wb") as output_file:
  pickle.dump(model, output_file)

X_test.head()

user_features = np.array([[10000, 2, 1,
                               1, 2,10]])

A = pd.DataFrame(user_features, columns=['km_driven', 'fuel', 'seller_type', 'transmission', 'past_owners', 'age'])

z_predict = model.predict(A)

print(type(model))

print(model)

!pip install joblib

import joblib

model = joblib.load('/content/carpriceprediction.pkl')

z_predict = model.predict(A)

print(z_predict)