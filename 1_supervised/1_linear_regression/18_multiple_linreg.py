from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load the data from the california house-prices dataset
california_data = fetch_california_housing()
x = california_data['data']
y = california_data['target']

# Make and fit the linear regression model
model = LinearRegression()
model.fit(X=x, y=y)

# Make a prediction using the model
sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
# TODO: Predict housing price for the sample_house
prediction = model.predict(X=sample_house)

print(prediction)
