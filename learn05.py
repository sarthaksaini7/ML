# logistical regression code
# regression algorithm does classification
# calculates the probability of belonging to a particular class

from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris["data"][:, 3:]
# calculate if a flower is iris virginica or not
# set the boolean value as an integer 1 or 0
y = (iris["target"] == 2).astype(int)

# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x, y)
example = clf.predict(([[3.6]]))
print(example)

# Using matplotlib to plot the visualisation
x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new, y_prob[:, 1], "g-", label="virginica")
plt.show()

# print(y_prob)
# print(x_new)
# print(x)
# print(y)
# print(list(iris.keys()))
# print(iris['data'])
# print(iris['target'])
# print(iris['DESCR'])
