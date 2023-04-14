import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Load the data into a pandas dataframe
data = {
    'year': [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
             2013, 2014, 2015, 2016, 2017, 2018],
    'vessels_arrived': [3612, 2857, 4087, 4233, 4339, 4232, 1195, 4062, 4032, 3883, 4139, 5117, 5366, 4811, 4456, 4067,
                        4332, 4134, 3976, 4264, 4993, 4993, 4879, 4874],
    'total_cargo_handled': [12568, 13659, 17569, 21569, 20365, 23569, 26955, 30659, 35695, 39587, 43598, 40594, 40225,
                            42365, 48778, 61240, 65069, 64970, 66243, 74410, 77579, 86519, 93857, 104935],
    'total_container_traffic_TEUs_000': [459, 853, 1453, 1269, 1745, 1846, 1922, 1965, 2056, 2154, 2369, 2694, 3068,
                                         3265, 3464, 4137, 4263, 4187, 4306, 4908, 5185, 5735, 6209, 7047],
    'total_revenue_rs_million': [8426, 9548, 10698, 11569, 14695, 13695, 13695, 15698, 17846, 17569, 17894, 18459,
                                 20369, 21569, 23331, 30377, 30377, 37120, 35200, 36776, 40164, 42994, 42514, 50124],
    'operating_expenditure': [7864, 8624, 9658, 10265, 12691, 15269, 14589, 16958, 17595, 18694, 19587, 20547, 21569,
                              22697, 23331, 28279, 30377, 37120, 35200, 36776, 40164, 42994, 42514, 50224],
    'y': [16492, 17476, 19033, 18777, 18930, 19344, 18561, 17910, 13936, 13233, 13527, 13660, 13667, 13715, 13367,
          12828, 11008, 10200, 9886, 9598, 9550, 9651, 9377, 10210]}

X = pd.DataFrame(data)
X = X.drop(['y'], axis=1)
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# random forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
dump(rf, 'trained_models/regression/random_forest.joblib')

# logistic regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
dump(lr, 'trained_models/regression/logistic_regression.joblib')

# Naive bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
dump(nb, 'trained_models/regression/naive_bayes.joblib')

# Decision tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
dump(dt, 'trained_models/regression/decision_tree.joblib')

# SVM
svm = SVR(kernel='linear')
svm.fit(X_train, y_train)

y_pred_svm = svm.predict(X_test)
dump(svm, 'trained_models/regression/svm.joblib')

print('MSE and R2 scores:')
print('Random Forest:', mean_squared_error(y_test, y_pred_rf), r2_score(y_test, y_pred_rf))
print('Logistic Regression:', mean_squared_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr))
print('Naive Bayes:', mean_squared_error(y_test, y_pred_nb), r2_score(y_test, y_pred_nb))
print('Decision Tree:', mean_squared_error(y_test, y_pred_dt), r2_score(y_test, y_pred_dt))
print('SVM:', mean_squared_error(y_test, y_pred_svm), r2_score(y_test, y_pred_svm))

new_data = [2023, 5000, 200000, 10000, 50000, 60000] # Example data for one year
new_target = rf.predict([new_data])
print("Prediction for y:", new_target)
