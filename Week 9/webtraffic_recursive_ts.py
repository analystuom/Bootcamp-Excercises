import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

data = pd.read_csv('Time-series-datasets/web-traffic.csv')
data["date"] = pd.to_datetime(data["date"], format="mixed")
data = data.sort_values(by="date", ascending=True).reset_index(drop=True)


def create_ts_data(data, window_size):
    i = 1
    while i < window_size:
        data["users_{}".format(i)] = data["users"].shift(-i)
        i += 1
    data["target"] = data["users"].shift(-i)
    data.dropna(axis=0, inplace=True)
    return data


data = create_ts_data(data, window_size=5)

x = data.drop(["target","date"], axis=1, inplace=False)
y = data["target"]

num_samples = len(data)
train_ratio = 0.8
x_train = x[:int(num_samples * train_ratio)]
y_train = y[:int(num_samples * train_ratio)]
x_test = x[int(num_samples * train_ratio):]
y_test = y[int(num_samples * train_ratio):]

models = {
    "Linear Regression" :LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=50)}

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Model {}".format(name))
    print("Mean squared error: {}".format(mean_squared_error(y_test, y_pred)))
    print("Mean absolute error: {}".format(mean_absolute_error(y_test, y_pred)))
    print("R2 score: {}".format(r2_score(y_test, y_pred)))

'''
Mục tiêu: Predict users dựa theo time-series data
Cả 2 mô hình tuyến tính và phi tuyến đều không perform tốt với dataset này, vậy nên có thể cân nhắc approach 
khác trong tương lai
'''










