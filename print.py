
# Load libraries
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = "iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(data, names=names)

array = dataset.values
X = array[1:,0:4]
y = array[1:,4]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

logisticRegression = LogisticRegression().fit(X_train, y_train)

print(logisticRegression.score(X_test, y_test))

