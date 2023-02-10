import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the data into a pandas dataframe
file_path = r"C:\Users\kanem\Documents\ML_Bike_Rental\data\raw\daily-bike-share.csv"
df = pd.read_csv(file_path)

df.columns
# Define the features and target
X = df[['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday','weathersit','temp','atemp','hum','windspeed',]]
y = df['rentals']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define the linear regression model
model = LinearRegression()

# fit model to data
model.fit(X_train, y_train)

# test model on test data

test_score = model.score(X_test, y_test)
print("R2 test score", test_score)


