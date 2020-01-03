import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

""" 
    Obj: predict the grade #3 with the other 5 five values
            G1, G2, studytime, failures, absences -> G3

    DataSet from: https://archive.ics.uci.edu/ml/datasets/Student+Performance
    Tutorial from:  Python Machine Learning Tutorial #3 - Linear Regression p.1
                    Python Machine Learning Tutorial #3 - Linear Regression p.2
        Link: https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr
"""

#load data
data = pd.read_csv("student-data_set.csv", sep=";")
print(data.head())
data = data[ ["G1", "G2", "G3", "studytime", "failures", "absences"] ]
print(data.head())

# get data needed
predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#training
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# generate the linear regression (y = mx+b) with the train
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test) # the percent accuracy

print("\nacc :",accuracy ) # percent so relative
print("Coeficiente: ", linear.coef_ )    # m values, amount related to dimensional space 
print("Intercept: ", linear.intercept_ ) # b (intercept with y line)

#with the model, predict....
print("Predictions: \n")
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], "\t\t", x_test[i], "\t\t", y_test[i])
