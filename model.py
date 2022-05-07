import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def train_model(model, data):
    # test_data = pd.read_csv('test_feb2022.csv')
    train_x, train_y = data.drop(["RoomTemperature"], axis=1), data["RoomTemperature"]
    model.fit(train_x, train_y)
    predictions = model.predict(train_x)
    rms = mean_squared_error(train_y, predictions, squared=False)

    return rms
