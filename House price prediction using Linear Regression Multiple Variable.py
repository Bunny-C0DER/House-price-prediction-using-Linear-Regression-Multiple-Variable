#importing libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#loading dataset

dataset = pd.read_csv('house.csv')

#summarizing dataset

print(dataset.shape)
print(dataset.head(5))

#finding NA values from our dataset

dataset.columns[dataset.isna().any()]

#removing NA values from our dataset

dataset.LotFrontage = dataset.LotFrontage.fillna(dataset.LotFrontage.mean())

#rechecking for NA Values

dataset.columns[dataset.isna().any()]

#Mapping Salary data into Interger Values

dataset['MSZoning'] = dataset['MSZoning'].map({'RL': 1, 'RM':2, 'FV':3, 'RH':4, 'C (all)':5}).astype(int)

dataset['HouseStyle'] = dataset['HouseStyle'].map({'1Story': 1, '2Story':2, '1.5Fin':3, '1.5Unf':4, '2.5Fin':5, '2.5Unf':6, 'SLvl':7, 'SFoyer':8}).astype(int)

print(dataset.head)

#segregating dataset into X and Y

X=dataset.drop('SalePrice',axis='columns')
X
print(np.shape(X))

Y=dataset.SalePrice
Y

#training dataset using Linear Regression

model = LinearRegression()
model.fit(X,Y)

#predicting price for house

x = [[20,1,80,10000,2,7,5]]
result = model.predict(x)
print(result)