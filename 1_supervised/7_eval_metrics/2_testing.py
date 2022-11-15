from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Read in the data.
data = np.asarray(pd.read_csv('2_data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y.
X = data[:, 0:2]
y = data[:, 2]

# Use train test split to split your data
# Use a test size of 25% and a random state of 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Instantiate your decision tree model
model = DecisionTreeClassifier()

# Fit the model to the training data.
model.fit(X=X_train, y=y_train)

# Make predictions on the test data
y_pred = model.predict(X=X_test)

# Calculate the accuracy and assign it to the variable acc on the test data.
acc = accuracy_score(y_true=y_test, y_pred=y_pred)
print(acc)
