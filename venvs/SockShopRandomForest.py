from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization
from mpl_toolkits import mplot3d  # for data visualization_3D
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


dataset = pd.read_csv('Experiment_SockShop.csv')

# get full dataset
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5]].values
y = dataset.iloc[:, [7]].values
# 0-Order API Concurrency
# 1-Carts API Concurrency
# 2-Order Cores
# 3-Order DB Cores
# 4-Carts Cores
# 5-Carts DB Cores

# 7-Average latency

# Standardization of data


scalerX = MinMaxScaler()
scalerY = MinMaxScaler()
X = scalerX.fit_transform(X)
y = scalerY.fit_transform(y)

print('Standardization done')
print(X)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.3, random_state=42)

# change the shape of y to (n_samples, )
yTrain = yTrain.ravel()

from sklearn.ensemble import RandomForestRegressor

# fit model no training data
model = RandomForestRegressor(n_estimators=20, random_state=0)
model.fit(XTrain, yTrain)

# Calculate errors
yTestPredict = model.predict(XTest)
print('eufghe3uf', yTest)
mse = mean_squared_error(yTest, yTestPredict, squared=True)
rmse = mean_squared_error(yTest, yTestPredict, squared=False)
mae = mean_absolute_error(yTest, yTestPredict)
mape = mean_absolute_percentage_error(yTest, yTestPredict)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The root Mean Square Error (RMSE) on test set: {:.4f}".format(rmse))
print("The mean absolute error on test set: {:.4f}".format(mae))
print("The mean absolute percentage error on test set: {:.4f}".format(mape))
errors = [['mean squared error (MSE)', mse], ['root Mean Square Error (RMSE)', rmse], ['mean absolute error', mae],
          ['mean absolute percentage error', mape]]

print(model.get_params(deep=True))

# prediction part
Order_API_Concurrency = 5
Carts_API_Concurrency = 5
Order_Cores = 0.2
Order_DB_Cores = 0.2
Carts_Cores = 0.2
Carts_DB_Cores = 0.2

orderCoresArray = np.arange(0.2, 0.7, 0.2)
orderDBCoresArray = np.arange(0.2, 0.7, 0.2)
cartsCoresArray = np.arange(0.2, 0.7, 0.2)
cartsDBCoresArray = np.arange(0.2, 0.7, 0.2)

newXArray = []
for i in orderCoresArray:
    for j in orderDBCoresArray:
        for k in cartsCoresArray:
            for l in cartsDBCoresArray:
                newx = [Order_API_Concurrency, Carts_API_Concurrency, i, j, k, l]
                newXArray.append(newx)

# new_X = [Order_API_Concurrency, Carts_API_Concurrency, Order_Cores, Order_DB_Cores, Carts_Cores, Carts_DB_Cores]
# print('X value ', new_X)

predicted_y = scalerY.inverse_transform([model.predict(scalerX.fit_transform(newXArray))])
print('Predicted y ', predicted_y)

for p in range(len(newXArray)):
    newXArray[p].append(predicted_y[0][p])

df = pd.DataFrame(newXArray)
df1 = pd.DataFrame(predicted_y)
df2 = pd.DataFrame(errors)

## save to xlsx file

filepath = 'my_excel_fileRandomForest.xlsx'

with pd.ExcelWriter(filepath) as writer:
    df.to_excel(writer, sheet_name='input and output')
    df1.to_excel(writer, sheet_name='output')
    df2.to_excel(writer, sheet_name='Accuracy Matrix')
