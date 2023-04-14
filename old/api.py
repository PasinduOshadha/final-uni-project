from flask import Flask, request
from flask_cors import CORS
import json
from joblib import dump, load
from datetime import datetime, timedelta
import pandas as pd

from statsmodels.tsa.arima.model import ARIMAResults

app = Flask(__name__)
CORS(app)

df = pd.read_csv('data.csv', index_col='Date', parse_dates=True)
df = df.dropna()

loaded = ARIMAResults.load('arima_model.pkl')


@app.route('/regression', methods=['GET', 'POST'])
def regression():
    loaded_model = load('regression_model.joblib')

    year = int(request.json['year'])
    vessels_arrived = int(request.json['vessels_arrived'])
    total_cargo_handled = int(request.json['total_cargo_handled'])
    total_container_traffic_TEUs_000 = int(request.json['total_container_traffic_TEUs_000'])
    total_revenue_rs_million = int(request.json['total_revenue_rs_million'])
    operating_expenditure = int(request.json['operating_expenditure'])

    new_data = [year, vessels_arrived, total_cargo_handled, total_container_traffic_TEUs_000, total_revenue_rs_million,
                operating_expenditure]
    new_target = loaded_model.predict([new_data])

    return_str = '{ "y" : ' + str(new_target) + ' }'

    # print(return_str)

    return json.loads(return_str)


@app.route('/arima', methods=['GET', 'POST'])
def arima():
    no_of_days = int(request.json['no_of_days']) - 1

    today = datetime.now()
    n_days = today + timedelta(days=no_of_days)
    print(today.strftime("%Y.%m.%d"), n_days.strftime("%Y.%m.%d"))
    index_future_dates = pd.date_range(start=today.strftime("%Y.%m.%d"), end=n_days.strftime("%Y.%m.%d"))
    # print(index_future_dates)
    pred = loaded.predict(start=len(df), end=len(df) + no_of_days, typ='levels').rename('ARIMA Predictions')
    # print(comp_pred)
    pred.index = index_future_dates

    return_str = '{ '
    for x in range(len(pred)):
        if x == len(pred) - 1:
            return_str += '"' + str(x + 1) + '" : ' + str(round(pred[x], 2))
        else:
            return_str += '"' + str(x + 1) + '" : ' + str(round(pred[x], 2)) + ','

    return_str += '}'

    print(return_str)

    return json.loads(return_str)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=True)
