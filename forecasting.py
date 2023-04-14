import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMAResults

# Load the daily time series dataset
data = pd.read_csv('data/data.csv', index_col='ds')
data.index = pd.to_datetime(data.index)

# Split the dataset into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size]
test_data = data.iloc[train_size:]

# LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

lstm_model = create_lstm_model()
lstm_history = lstm_model.fit(train_data.values.reshape(-1, 1, 1),
                              train_data.values,
                              epochs=100,
                              batch_size=64,
                              verbose=0)

lstm_predictions = lstm_model.predict(test_data.values.reshape(-1, 1, 1))
lstm_mae = mean_absolute_error(test_data.values, lstm_predictions)
print('LSTM MAE:', lstm_mae)
lstm_model.save('trained_models/forecasting/lstm_model.h5')

# ARIMA model
arima_model = ARIMA(train_data.values, order=(2, 1, 3))
arima_model_fit = arima_model.fit()
arima_predictions = arima_model_fit.forecast(len(test_data))
arima_mae = mean_absolute_error(test_data.values, arima_predictions)
print('ARIMA MAE:', arima_mae)
arima_model_fit.save('trained_models/forecasting/arima_model.pkl')

# FBP model
fbp_model = Prophet()
fbp_model.fit(train_data.reset_index())
fbp_forecast = fbp_model.predict(test_data.reset_index())
fbp_predictions = fbp_forecast['yhat'].values
fbp_mae = mean_absolute_error(test_data.values, fbp_predictions)
print('FBP MAE:', fbp_mae)
fbp_model.save('trained_models/forecasting/fbp_model.h5')


############# Test

# Load the trained models
lstm_model = load_model('trained_models/forecasting/lstm_model.h5')
arima_model = ARIMAResults.load('trained_models/forecasting/arima_model.pkl')

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define the forecast horizon
horizon = 30  # number of days to forecast

# Make predictions with the LSTM model
lstm_input = scaled_data[-horizon:]
# lstm_input = lstm_input.reshape((1, lstm_input.shape[0], lstm_input.shape[1]))
lstm_preds = lstm_model.predict(lstm_input)
lstm_preds = pd.DataFrame(lstm_preds, columns=['Value'], index=pd.date_range(start=data.index[-1], periods=horizon+2, closed='right')[1:])


x = lstm_preds.values.tolist()

for i in x:
    print(i[0])

# Make predictions with the ARIMA model
arima_preds = arima_model.forecast(steps=horizon)
arima_preds = pd.DataFrame(arima_preds[0], columns=['Value'], index=pd.date_range(start=data.index[-1], periods=horizon+1, closed='right')[1:])


# Combine the predictions into a single dataframe
preds = pd.concat([lstm_preds, arima_preds], axis=1)
preds.columns = ['LSTM', 'ARIMA']

print(preds)
