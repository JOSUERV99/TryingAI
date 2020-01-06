import pandas as pd
import numpy as np
import sklearn, pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style
import matplotlib.pyplot as pyplot

""" 
    Obj: predict the grade #3 with the other 5 five values
            G1, G2, studytime, failures, absences -> G3       (95.1891 %)

    DataSet from: https://archive.ics.uci.edu/ml/datasets/Student+Performance
    Tutorial from:  Python Machine Learning Tutorial #3 - Linear Regression p.1
                    Python Machine Learning Tutorial #3 - Linear Regression p.2
        Link: https://www.youtube.com/playlist?list=PLzMcBGfZo4-mP7qA9cagf68V06sko5otr
"""

#load data
def loadCSV(filename, headers, separator=";"):
        data = pd.read_csv("{}.csv".format(filename), sep=separator)
        print(data.head())
        data = data[ headers  ]
        print(data.head())
        return data

def saveModel(filename, model):
        with open("{}.pickle".format(filename), "wb") as f:
                pickle.dump(model, f)
        
def loadModel(filename):
        file = open("{}.pickle".format(filename), "rb")
        return pickle.load(file)

# get data needed
data = loadCSV("student-data_set", ["G1", "G2", "G3", "studytime", "failures", "absences"])
predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# TRAINING
# best = 0
# for i in range(500):
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
#         # generate the linear regression (y = mx+b) with the train
#         linear = linear_model.LinearRegression()
#         linear.fit(x_train, y_train)
#         accuracy = linear.score(x_test, y_test) # the percent accuracy
#         if accuracy > best:
#                 best = accuracy
#                 print("Until now {}".format(accuracy))
#                 saveModel("studentModel_{}_".format(accuracy*100), linear)
linear = loadModel("studentModel")

#print("\nacc :",accuracy ) # percent so relative
print("Coeficiente: ", linear.coef_ )    # m values, amount related to dimensional space 
print("Intercept: ", linear.intercept_ ) # b (intercept with y line)

#with the model, predict....
print("Predictions: \n")
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(predictions[i], "\t\t", x_test[i], "\t\t", y_test[i])
