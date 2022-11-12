import pandas as pd
from sklearn.linear_model import LinearRegression

# Assign the dataframe to this variable.
bmi_life_data = pd.read_csv('17_bmi_and_life_expectancy.csv')

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi_life_model.fit(X=bmi_life_data[['BMI']].values, y=bmi_life_data[['Life expectancy']].values)

# Make a prediction using the model
laos_life_exp = bmi_life_model.predict(X=[[21.07931]])

print(laos_life_exp)
