import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization
from mpl_toolkits import mplot3d  # for data visualization_3D
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


dataset = pd.read_csv('Experiment_SockShop.csv')

# calculate number of rows in tha dataset
nor = dataset['Order_Cores'].count()
print('Number of rows in the database ', nor)

# set for the values to train and test percentages
TrainPercentage = 0.7
TestPercentage = 0.3

# calculate  number of rows to test and train based on the percentage
numberOfRowsToTrain = nor * TrainPercentage
numberOfRowsToTest = nor * TestPercentage
print('Number of rows to train the database ', numberOfRowsToTrain)
print('Number of rows to test the database ', numberOfRowsToTest)

# get full dataset
X = dataset.iloc[:nor, [0, 1, 2, 3, 4, 5]].values
y = dataset.iloc[:nor, [7]].values
# 0-Order API Concurrency
# 1-Carts API Concurrency
# 2-Order Cores
# 3-Order DB Cores
# 4-Carts Cores
# 5-Carts DB Cores

# 7-Average latency

# Standardization of data
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

print('Standardization done')
print()

# shuffle full dataset
X, y = shuffle(X, y)

# get train dataset
XTrain = X[:int(numberOfRowsToTrain)]
yTrain = y[:int(numberOfRowsToTrain)]

# get test dataset
XTest = X[int(numberOfRowsToTrain):int(nor)]
yTest = y[int(numberOfRowsToTrain):int(nor)]

# change the shape of y to (n_samples, )
# yTrain = yTrain.ravel()

# train the model
from sklearn.svm import SVR

mseValues = []
rmseValues = []
maeValues = []
mapeValues = []
cGrid = []
degreeGrid = []
epsilonValues = []

regressor = linear_model.LogisticRegression()
regressor.fit(XTrain, yTrain)

# Calculate errors
yTestPredict = regressor.predict(XTest)
mse = mean_squared_error(yTest, yTestPredict, squared=True)
rmse = mean_squared_error(yTest, yTestPredict, squared=False)
mae = mean_absolute_error(yTest, yTestPredict)
mape = mean_absolute_percentage_error(yTest, yTestPredict)
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
print("The root Mean Square Error (RMSE) on test set: {:.4f}".format(rmse))
print("The mean absolute error on test set: {:.4f}".format(mae))
print("The mean absolute percentage error on test set: {:.4f}".format(mape))
print(regressor.get_params(deep=True))

# prediction part
Order_API_Concurrency = 5
Carts_API_Concurrency = 5
Order_Cores = 0.2
Order_DB_Cores = 0.2
Carts_Cores = 0.2
Carts_DB_Cores = 0.2

new_X = [Order_API_Concurrency, Carts_API_Concurrency, Order_Cores, Order_DB_Cores, Carts_Cores, Carts_DB_Cores]
print()
print('X value ', new_X)

predicted_y= sc_y.inverse_transform(regressor.predict(sc_X.fit_transform([new_X])))
print('Predicted y ',predicted_y)
