from flask import Flask, request
from flask_cors import CORS
import json
from joblib import dump, load
from datetime import datetime, timedelta
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.arima.model import ARIMAResults

app = Flask(__name__)
CORS(app)

# Load the daily time series dataset
data = pd.read_csv('data/data.csv', index_col='ds')
data.index = pd.to_datetime(data.index)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Load the trained models
lstm_model = load_model('trained_models/forecasting/lstm_model.h5')


@app.route('/regression', methods=['GET', 'POST'])
def regression():
    loaded_model = load('trained_models/regression/random_forest.joblib')  # you can change model as you like

    year = int(request.json['year'])
    vessels_arrived = int(request.json['vessels_arrived'])
    total_cargo_handled = int(request.json['total_cargo_handled'])
    total_container_traffic_TEUs_000 = int(request.json['total_container_traffic_TEUs_000'])
    total_revenue_rs_million = int(request.json['total_revenue_rs_million'])
    operating_expenditure = int(request.json['operating_expenditure'])

    new_data = [year, vessels_arrived, total_cargo_handled, total_container_traffic_TEUs_000, total_revenue_rs_million,
                operating_expenditure]
    new_target = loaded_model.predict([new_data])

    return_str = '{ "no_of_employees" : ' + str(new_target[0]) + ' }'

    # print(return_str)

    return json.loads(return_str)


@app.route('/forecasting', methods=['GET', 'POST'])
def forecasting():
    horizon = int(request.json['no_of_days'])

    # Make predictions with the LSTM model
    lstm_input = scaled_data[-horizon:]
    # lstm_input = lstm_input.reshape((1, lstm_input.shape[0], lstm_input.shape[1]))
    lstm_preds = lstm_model.predict(lstm_input)
    lstm_preds = pd.DataFrame(lstm_preds, columns=['Value'],
                              index=pd.date_range(start=data.index[-1], periods=horizon + 2, closed='right')[1:])
    pred = lstm_preds.values.tolist()

    return_str = '{ '
    for x in range(len(pred)):
        if x == len(pred) - 1:
            return_str += '"' + str(x + 1) + '" : ' + str(round(pred[x][0], 2))
        else:
            return_str += '"' + str(x + 1) + '" : ' + str(round(pred[x][0], 2)) + ','

    return_str += '}'

    print(return_str)

    return json.loads(return_str)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
