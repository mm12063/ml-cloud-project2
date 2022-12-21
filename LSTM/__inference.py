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
import VARS as _GLOBALS

today = date.today()
yesterday = today - timedelta(days=1)
n_years_ago = yesterday - relativedelta(years=_GLOBALS.NUM_YRS)
start_date = n_years_ago
end_date = yesterday
dates = pd.date_range(start_date,end_date,freq='B')
df1 = pd.DataFrame(index=dates)


def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data)
    test_set_size = int(np.round(0.2*data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)

    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]

    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]

    return [x_train, y_train, x_test, y_test]


input_dim = 1
hidden_dim = 32
num_layers = 5 
output_dim = 1

# Here we define our model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states! 
        out = self.fc(out[:, -1, :]) 
        # out.size() --> 100, 10
        return out

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
look_back = 5
seq_dim =look_back-1

def load_data_inference(stock, look_back):
    data_raw = stock.values # convert to numpy array
    # print(data_raw)
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data)
    test_set_size = int(1)
    train_set_size = data.shape[0] - (test_set_size)
    x_test = data[train_set_size:,-(look_back-1):]
    y_test = data[train_set_size:,-1,:]
    
    return [x_test, y_test]


def inference(symb):
    df_ibm = pd.read_csv(f"./Stocks/csv_{_GLOBALS.NUM_YRS}yrs/{symb}.csv", parse_dates=True, index_col=0)
    df_ibm = df1.join(df_ibm)
    df_ibm = df_ibm[['Close']]
    df_ibm = df_ibm.fillna(method='ffill')
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1, 1))

    x_train, y_train, x_test, y_test = load_data(df_ibm, look_back)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)

    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    PATH = f"./Model/model_{symb}.pt"
    model.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
    model.eval()
    model.to(device)

    x_test = x_test.to(device)
    y_test_pred = model(x_test)

    y_test_pred = scaler.inverse_transform(y_test_pred.cpu().detach().numpy())
    y_test = scaler.inverse_transform(y_test.cpu().detach().numpy())
    testScore = math.sqrt(mean_squared_error(y_test[:, 0], y_test_pred[:, 0]))

    num_days_to_predict = 3
    for day in range(num_days_to_predict):
        x_test_future, y_test_furture = load_data_inference(df_ibm, look_back)
        x_test_future = torch.from_numpy(x_test_future).type(torch.Tensor)
        y_predicted = model(x_test_future)
        y_predicted_inverse = scaler.inverse_transform(y_predicted.cpu().detach().numpy())
        y_test_pred = np.append(y_test_pred, y_predicted_inverse)
        df_ibm.loc[today + timedelta(days=day)] = y_predicted.item()

    return testScore, y_test, y_test_pred




