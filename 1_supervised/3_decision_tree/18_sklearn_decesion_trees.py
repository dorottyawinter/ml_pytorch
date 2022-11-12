from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = np.asarray(pd.read_csv('18_data.csv', header=None))

X = data[:, 0:2]
y = data[:, 2]

model = DecisionTreeClassifier()
model.fit(X=X, y=y)

y_pred = model.predict(X=X)

acc = accuracy_score(y_true=y, y_pred=y_pred)
print(acc)
