import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

# Load the data into a pandas dataframe
data = {'year': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
        'vessels_arrived': [3612, 2857, 4087, 4233, 4339, 4232, 1195, 4062, 4032, 3883, 4139, 5117, 5366, 4811, 4456, 4067, 4332, 4134, 3976, 4264, 4993, 4993, 4879, 4874],
        'total_cargo_handled': [12568, 13659, 17569, 21569, 20365, 23569, 26955, 30659, 35695, 39587, 43598, 40594, 40225, 42365, 48778, 61240, 65069, 64970, 66243, 74410, 77579, 86519, 93857, 104935],
        'total_container_traffic_TEUs_000': [459, 853, 1453, 1269, 1745, 1846, 1922, 1965, 2056, 2154, 2369, 2694, 3068, 3265, 3464, 4137, 4263, 4187, 4306, 4908, 5185, 5735, 6209, 7047],
        'total_revenue_rs_million': [8426, 9548, 10698, 11569, 14695, 13695, 13695, 15698, 17846, 17569, 17894, 18459, 20369, 21569, 23331, 30377, 30377, 37120, 35200, 36776, 40164, 42994, 42514, 50124],
        'operating_expenditure': [7864, 8624, 9658, 10265, 12691, 15269, 14589, 16958, 17595, 18694, 19587, 20547, 21569, 22697, 23331, 28279, 30377, 37120, 35200, 36776, 40164, 42994, 42514, 50224],
        'y': [16492, 17476, 19033, 18777, 18930, 19344, 18561, 17910, 13936, 13233, 13527, 13660, 13667, 13715, 13367, 12828, 11008, 10200, 9886, 9598, 9550, 9651, 9377, 10210]}
df = pd.DataFrame(data)

# Split the data into features and target
X = df.drop('y', axis=1)
y = df['y']

print(len(X), len(y))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the linear regression model
reg = LinearRegression().fit(X_train, y_train)

# Use the trained model to make predictions on the test data
predictions = reg.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

new_data = [2023, 5000, 200000, 10000, 50000, 60000] # Example data for one year
new_target = reg.predict([new_data])
print("Prediction for y:", new_target)

# Save the model to a file
dump(reg, 'regression_model.joblib')

# Load the saved model
loaded_model = load('regression_model.joblib')



# for fine tuning model
# Create the model
reg = Ridge()

# Define a range of hyperparameters to search over
hyperparameters = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

# Use grid search to find the best hyperparameters
reg_cv = GridSearchCV(reg, hyperparameters, cv=5)
reg_cv.fit(X_train, y_train)

# Print the best hyperparameters
print("Best alpha:", reg_cv.best_params_)
