from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge

# gettin dataset
dataset = load_boston()

# neighbors way
x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target)
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)
print("KNN Score: ", knn.score(x_test, y_test))

#linear regression way
rl = LinearRegression()
rl.fit(x_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=True)
print("LR Score: ", rl.score(x_test, y_test))

# rideg way
ridge = Ridge()
ridge.fit(x_train, y_train)
ridge.score(x_test, y_test)
print("Ridge Score: ", knn.score(x_test, y_test))

