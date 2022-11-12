import pandas as pd
from sklearn.linear_model import Lasso

# Assign the data to predictor and outcome variables
train_data = pd.read_csv('26_data.csv', header=None)
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, -1].values

# TODO: Create the linear regression model with lasso regularization.
lasso_reg = Lasso()

# TODO: Fit the model.
lasso_reg.fit(X=X, y=y)

# TODO: Retrieve and print out the coefficients from the regression model.
reg_coef = lasso_reg.coef_
print(reg_coef)
