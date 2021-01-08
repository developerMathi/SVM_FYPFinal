# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization

dataset = pd.read_csv('Test_Results.csv')
##print(dataset)

X = dataset.iloc[:1500,[1,2,4]].values
print(X[0:5,:])
X_test = dataset.iloc[1500:1750,[1,2,4]].values

##1-heap size
##2-concurrency
##4-prime number
##
##for i  in Xx:
##    li=[64,1]
##    li.append(i[0])
##    print(li)
##print(Xx)
y = dataset.iloc[:1500,[6]].values
print(y[0:5,:])
y_test = dataset.iloc[1500:1750,[6]].values



##print (X)
##print (y)


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
X_test = sc_X.transform(X_test)
print(X[0:5,:])

y = sc_y.fit_transform(y)
y_test = sc_y.transform(y_test)
print(y[0:5,:])

##print (y)

##print(X,y)

##format_train_y=[]
##for n in y:
##    format_train_y.append(n[0])
##print(y)
y = y.ravel()



from sklearn.svm import SVR
regressor = SVR()
##from sklearn.linear_model import LinearRegression
##regressor = LinearRegression()
regressor.fit(X,y)
##y_predicted = regressor.predict(X_test)
accuracy = regressor.score(X_test,y_test)
print('accuracy =',accuracy*100,'%')

y_pred = regressor.predict([[64,500,100003]])
print("X value is [64,500,100003] predicted Y is ",y_pred)
y_pred = sc_y.inverse_transform(y_pred)
print(dataset.iloc[[4],[1,2,4]].values)
print(y_pred)


X_grid = np.arange(1, 500, 1)
recreatedGrid=[]
for i in X_grid:
    li=[0,0,0]
    li[0]=64
    li[1]=i
    li[2]=100003
    recreatedGrid.append(li)
    # y_pred = regressor.predict([li])
    # y_pred = sc_y.inverse_transform(y_pred)
print(recreatedGrid)
    # print(y_pred)
        
##print(X_grid)

##plt.scatter(X[:1750,[1]], y,color = 'red')
plt.plot(X_grid, regressor.predict(recreatedGrid), color = 'blue')
plt.xlabel('concurrency')
plt.ylabel('Latency')
plt.show()


##
##X_grid = X_grid.reshape((len(X_grid), 1))
##plt.scatter(X, y,color = 'red')
##plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
##plt.title('Truth or Bluff (Support Vector Regression Model(High Resolution))')
##plt.xlabel('Position level')
##plt.ylabel('Salary')
##plt.show()
