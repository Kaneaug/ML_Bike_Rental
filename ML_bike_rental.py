import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset into a pandas dataframe
file_path = r'C:\Users\kanem\Documents\ML_Bike_Rental\data\daily-bike-share.csv'
df = pd.read_csv(file_path)

# Split the data into features and target
X = df.drop(['dteday', 'rentals'], axis=1)
y = df['rentals']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Evaluate the model performance using mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
