import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('Time-series-datasets/bike-sharing-dataset.csv')
data["date_time"] = pd.to_datetime(data["date_time"], format="mixed")
data = data[["date_time", "temp", "windspeed", "users"]]


# fig, ax = plt.subplots()
# ax.plot(data["date_time"], data["users"])
# ax.set_xlabel("Date")
# ax.set_ylabel("User")
# plt.show()

# Tạo thêm cả Exogenous variable là temp và windspeed
def create_ts_data(data, window_size, target_size):
    i = 1
    while i < window_size:
        data["users_{}".format(i)] = data["users"].shift(-i)
        i += 1
    i = 0
    while i < target_size:
        data["target_{}".format(i)] = data["users"].shift(-i - window_size)
        data["temp_{}".format(i)] = data["temp"].shift(-i - window_size)
        data["windspeed_{}".format(i)] = data["windspeed"].shift(-i - window_size)
        i += 1
    data.dropna(axis=0, inplace=True)
    return data


window_size = 5
target_size = 3
data = create_ts_data(data, window_size, target_size)


num_samples = len(data)
train_size = 0.8
targets = ["target_{}".format(i) for i in range(target_size)]
x = data.drop(["date_time", "users"], axis=1, inplace=False)
y = data[targets]
x_train = x[:int(num_samples * train_size)]
y_train = y[:int(num_samples * train_size)]
x_test = x[int(num_samples * train_size):]
y_test = y[int(num_samples * train_size):]

models = [LinearRegression() for i in range(target_size)]
for i, model in enumerate(models):
    model.fit(x_train, y_train["target_{}".format(i)])

for i, model in enumerate(models):
    y_predict = model.predict(x_test)
    rmse = mean_squared_error(y_test["target_{}".format(i)], y_predict)
    mae = mean_absolute_error(y_test["target_{}".format(i)], y_predict)
    r2 = r2_score(y_test["target_{}".format(i)], y_predict)
    print(f"RMSE: {rmse}", f"MAE: {mae}", f"R2: {r2}")