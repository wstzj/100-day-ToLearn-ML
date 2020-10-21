import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1:2].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

# Y_pred = regressor.predict(X_test)
# print(Y_pred)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.show()
