import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

# Ignore harmless warnings
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv', index_col='Date', parse_dates=True)
df = df.dropna()
print('Shape of data', df.shape)
print(df.head())


def adf_test(dataset):
    dftest = adfuller(dataset, autolag='AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t", key, ": ", val)


adf_test(df['y'])

stepwise_fit = auto_arima(df['y'], suppress_warnings=True)

print(stepwise_fit.summary())
print(df.shape)
train = df.iloc[:-30]
test = df.iloc[-30:]
print(train.shape, test.shape)
print(test.iloc[0], test.iloc[-1])

result_list = []

for i in range(4):
    for j in range(4):
        for k in range(4):
            try:
                model = ARIMA(train['y'], order=(i, j, k))
                model = model.fit()
                model.summary()

                start = len(train)
                end = len(train) + len(test) - 1

                pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA predictions')
                # pred.index=index_future_dates
                pred.plot(legend=True)
                test['y'].plot(legend=True)

                rmse = sqrt(mean_squared_error(pred, test['y']))
                print(rmse, str(i) + ' ' + str(j) + ' ' + str(k))
                result_list.append((rmse, str(i) + ' ' + str(j) + ' ' + str(k)))
            except:
                print(i, j, k, " model not working")

sorted = sorted(result_list, key=lambda tup: tup[0])
print("best model ", sorted[0])

model = ARIMA(train['y'], order=(2, 1, 3))
model = model.fit()
model.summary()

start = len(train)
end = len(train) + len(test) - 1

pred = model.predict(start=start, end=end, typ='levels').rename('ARIMA predictions')
# pred.index=index_future_dates
pred.plot(legend=True)
test['y'].plot(legend=True)

rmse = sqrt(mean_squared_error(pred, test['y']))
print(rmse)

model.save('arima_model.pkl')

index_future_dates = pd.date_range(start='2021-01-01', end='2021-01-31')
# print(index_future_dates)
pred = model.predict(start=len(df), end=len(df) + 30, typ='levels').rename('ARIMA Predictions')
# print(comp_pred)
pred.index = index_future_dates
print(pred)
