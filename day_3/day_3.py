import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()

X[:, 3] = labelencoder.fit_transform(X[:, 3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

regression = LinearRegression()
regression.fit(X_train, Y_train)

plt.plot(X_train, regression.predict(X_train), color='blue')
plt.show()

plt.plot(X_test, regression.predict(X_test), color='blue')
plt.show()
