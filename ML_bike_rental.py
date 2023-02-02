import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data into a pandas dataframe
file_path = r"C:\Users\kanem\Documents\ML_Bike_Rental\data\raw\daily-bike-share.csv"
df = pd.read_csv(file_path)

# Define the features and target
features = ['temp', 'hum', 'windspeed', 'weekday', 'workingday', 'weathersit']
target = 'rentals'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Train the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Use the model to make predictions on the test data
y_pred = reg.predict(X_test)

# Calculate the mean squared error (MSE) and mean absolute error (MAE) as test scores
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the test scores
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
