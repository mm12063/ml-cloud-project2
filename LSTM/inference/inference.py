import csv
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from minio import Minio

minio_client = Minio(
    "172.21.203.247:9000",
    access_key="minio",
    secret_key="minio123",
    secure=False
)
minio_bucket = "mlpipeline"


ROOT_PATH = '/home/app/'
NUM_YRS = 10

today = date.today()
yesterday = today - timedelta(days=1)
n_years_ago = yesterday - relativedelta(years=NUM_YRS)
start_date = n_years_ago
end_date = yesterday
dates = pd.date_range(start_date, end_date, freq='B')
df1 = pd.DataFrame(index=dates)

input_dim = 1
hidden_dim = 32
num_layers = 5
output_dim = 1

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
look_back = 5

def load_data(stock, look_back):
    data_raw = stock.values
    data = []

    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def load_data_inference(stock, look_back):
    data_raw = stock.values
    data = []

    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(1)
    train_set_size = data.shape[0] - (test_set_size)
    x_test = data[train_set_size:, -(look_back - 1):]
    y_test = data[train_set_size:, -1, :]

    return [x_test, y_test]


def get_rmse(symb):
    rmse_file_name = f"{symb}_RMSE.csv"
    rmse_loc = f'/home/app/RMSEs/{rmse_file_name}'
    try:
        minio_client.fget_object(minio_bucket, f'mlpipeline/rmses/{rmse_file_name}', rmse_loc)
    except:
        print(f"Couldn't load RMSE for: {symb}")

    with open(rmse_loc, mode='r') as infile:
        reader = csv.reader(infile)
        mydict = {f'{rows[0]}': rows[1] for rows in reader}

    return mydict[symb]


def get_inference(symb):
    csv_loc = f'/home/app/CSVs/{symb}.csv'
    model_loc = f'/home/app/models/model_{symb}.pt'
    try:
        minio_client.fget_object(minio_bucket, f'mlpipeline/stock-data/{symb}.csv', csv_loc)
        minio_client.fget_object(minio_bucket, f'mlpipeline/models/model_{symb}.pt', model_loc)
    except:
        print(f"Couldn't load data for: {symb}")

    df = pd.read_csv(csv_loc, parse_dates=True, index_col=0)
    df = df1.join(df)
    df = df[['Close']]
    df = df.fillna(method='ffill')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    x_train, y_train, x_test, y_test = load_data(df, look_back)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    model.load_state_dict(torch.load(model_loc))
    model.eval()

    y_test_pred = model(x_test)

    y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
    y_test = scaler.inverse_transform(y_test.cpu().detach().numpy())
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))

    dates = df[len(df) - len(y_test):].index
    dates = dates.strftime("%m/%d/%Y").tolist()

    num_days_to_predict = 3
    for day in range(num_days_to_predict):
        x_test_future, y_test_furture = load_data_inference(df, look_back)
        x_test_future = torch.from_numpy(x_test_future).type(torch.Tensor)
        y_predicted = model(x_test_future)
        y_predicted_inverse = scaler.inverse_transform(y_predicted.cpu().detach().numpy())
        y_test_pred = np.append(y_test_pred, y_predicted_inverse)
        df.loc[today + timedelta(days=day)] = y_predicted.item()

    return testScore, y_test, y_test_pred, dates

