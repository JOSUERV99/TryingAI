import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# obtenemos el dataset
dataset = load_iris()

# separamos los datos para prueba y de entrenamiento
x_train, x_test, y_train, y_test = train_test_split(dataset['data'], dataset['target'])

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
score = knn.score(x_test, y_test)
print("Score: ",score)

test = [[1.2, 3.4, 5.6, 7.8]]
result = int(knn.predict(test))
targets = dataset.target_names
print(test, targets[result])
