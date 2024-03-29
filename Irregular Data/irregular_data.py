import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import linear_model, preprocessing

def readData(filename):
    data = pd.read_csv(filename)
    return data

def preprocessingData(data):
    le = preprocessing.LabelEncoder()
    buying = le.fit_transform(list(data["buying"]))
    maint = le.fit_transform(list(data["maint"]))
    door = le.fit_transform(list(data["door"]))
    persons = le.fit_transform(list(data["persons"]))
    lug_boot = le.fit_transform(list(data["lug_boot"]))
    safety = le.fit_transform(list(data["safety"]))
    class_ = le.fit_transform(list(data["class"]))
    
    X = list(zip(buying, maint, door, persons, lug_boot, safety))
    y = list(class_)

    return X, y

predict = "class"
data = readData("car.data")
X, y = preprocessingData(data)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
model = KNeighborsClassifier(n_neighbors=9)
model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
predicted = model.predict(x_test)

print(accuracy) 
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(x_test)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("n:", n)