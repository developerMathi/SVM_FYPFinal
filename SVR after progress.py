import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data visualization
from mpl_toolkits import mplot3d  # for data visualization_3D
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.utils import check_array
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


dataset = pd.read_csv('Test_Results.csv')

# calculate number of rows in tha dataset
nor = dataset['Cores'].count()
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
X = dataset.iloc[:nor, [1, 2, 3, 4]].values
y = dataset.iloc[:nor, [6]].values
# 1-heap size
# 2-concurrency
# 3-cores
# 4-prime number
# 6-Average latency

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
yTrain = yTrain.ravel()

# train the model
from sklearn.svm import SVR

mseValues = []
rmseValues = []
maeValues = []
mapeValues = []
cGrid = []
degreeGrid = []
epsilonValues = []

cs = np.arange(0.1, 20, 0.5)
for i in cs:
    regressor = SVR(kernel='rbf', C=i)
    regressor.fit(XTrain, yTrain)

    yTestPredict = regressor.predict(XTest)
    mse = mean_squared_error(yTest, yTestPredict, squared=True)
    rmse = mean_squared_error(yTest, yTestPredict, squared=False)
    mae = mean_absolute_error(yTest, yTestPredict)
    mape = mean_absolute_percentage_error(yTest, yTestPredict)
    mseValues.append(mse)
    rmseValues.append(rmse)
    maeValues.append(mae)
    mapeValues.append(mape)
    cGrid.append(regressor.C)
    degreeGrid.append(regressor.degree)
    epsilonValues.append(regressor.epsilon)
    print('C value is ', regressor.C)
    print('Kernal is ', regressor.kernel)
    print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
    print("The root Mean Square Error (RMSE) on test set: {:.4f}".format(rmse))
    print("The mean absolute error on test set: {:.4f}".format(mae))
    print("The mean absolute percentage error on test set: {:.4f}".format(mape))
    print()
    print()

# regressor = SVR(kernel='rbf',C=5.0, epsilon=0.1)
regressor = SVR()
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

# plt.plot(degreeGrid, mseValues, color='blue')
# plt.xlabel('degree values')
# plt.ylabel('Mean square error values')
# plt.title('kernel = poly')
# plt.show()
#
# plt.plot(degreeGrid, rmseValues, color='red')
# plt.xlabel('degree values')
# plt.ylabel('Root mean square error values')
# plt.title('kernel = poly')
# plt.show()
#
# plt.plot(degreeGrid, maeValues, color='green')
# plt.xlabel('degree values')
# plt.ylabel('Mean absolute error values')
# plt.title('kernel = poly')
# plt.show()

plt.plot(cGrid, mapeValues, color='green')
plt.xlabel('C values')
plt.ylabel('Mean absolute percentage error values')
plt.title('kernel = rbf')
plt.show()

listGridc = np.arange(0.2, 2.2, 0.1)
recreatedGridc = []
for i in listGridc:
    li = [0, 0, 0, 0]
    li[0] = 1024
    li[1] = 500
    li[2] = i
    li[3] = 100003
    recreatedGridc.append(li)

newXnormalizedc = sc_X.fit_transform(recreatedGridc)
# yPred=sc_y.inverse_transform(regressor.predict(newXnormalized))
# print(yPred)
plt.plot(listGridc, sc_y.inverse_transform(regressor.predict(newXnormalizedc)), color='orange')
plt.xlabel('cores')
plt.ylabel('Latency')
plt.show()

listGridm = np.arange(1, 1025, 20)
recreatedGridm = []
for i in listGridm:
    li = [0, 0, 0, 0]
    li[0] = i
    li[1] = 500
    li[2] = 2
    li[3] = 100003
    recreatedGridm.append(li)

newXnormalizedm = sc_X.fit_transform(recreatedGridm)
plt.plot(listGridm, sc_y.inverse_transform(regressor.predict(newXnormalizedm)), color='orange')
plt.xlabel('Memory size')
plt.ylabel('Latency')
plt.show()

listGridcon = np.arange(100, 501, 10)
recreatedGridcon = []
for i in listGridcon:
    li = [0, 0, 0, 0]
    li[0] = 1024
    li[1] = i
    li[2] = 2
    li[3] = 100003
    recreatedGridcon.append(li)

newXnormalizedcon = sc_X.fit_transform(recreatedGridcon)
plt.plot(listGridcon, sc_y.inverse_transform(regressor.predict(newXnormalizedcon)), color='orange')
plt.xlabel('Concurrency')
plt.ylabel('Latency')
plt.show()

lisMem3D = np.linspace(0, 1024, 150)
lisCon3D = np.linspace(100, 500, 150)
lisCore3D = np.linspace(0, 2, 150)
j = 0;
recreated3D = []
while j < 150:
    li = [0, 0, 0, 0]
    li[0] = lisMem3D[j]
    li[1] = 200
    li[2] = lisCore3D[j]
    li[3] = 100003
    j += 1
    recreated3D.append(li)

newXnormalizedcon = sc_X.fit_transform(recreated3D)
ax = plt.axes(projection="3d")
ax.plot3D(lisCore3D, lisMem3D,  sc_y.inverse_transform(regressor.predict(newXnormalizedcon)))
plt.xlabel('Cores')
plt.ylabel('Memory')
plt.show()

# print(sc_y.inverse_transform(regressor.predict(sc_X.fit_transform([[2, 500, 1024, 100003]]))))
#
# ax = plt.axes(projection="3d")
# xxx = np.linspace(0, 2, 20)
# yyy = np.linspace(0, 1024, 20)
#
#
# def z_function(xxx, yyy):
#     return np.int(regressor.predict([[yyy, 500, xxx, 100003]])[0])
#
# def z_functionnor(xxx, yyy):
#     print([[yyy, 500, xxx, 100003]])
#     return sc_X.fit_transform([[yyy, 500, xxx, 100003]])
#
#
# def z_functions(xxx, yyy):
#     return xxx ** 2 + yyy ** 2
#
#
# print(z_function(2, 1024))
# print(z_function(1, 512))
# print(z_function(0, 0))
# print(z_function(0, 1024))
# print(z_functions(2, 512))
# # X, Y = np.meshgrid(x, y)
# # print(X, Y)
# ax.plot3D(xxx, yyy, z_function(xxx, yyy))
#
# plt.show()
