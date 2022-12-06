import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

from pandas import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

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
df1=pd.DataFrame(index=dates)


def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    # print(data_raw)
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data)
    print(data.shape)
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
look_back = 5
seq_dim = look_back-1
num_epochs = 100

model_save_dir = f'{_GLOBALS.ROOT_PATH}Model/'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

try:
  os.remove(f"{model_save_dir}RMSE.csv")
except OSError:
  pass

rmsfile = open(f"{model_save_dir}RMSE.csv", "w")
for i, symb in enumerate(_GLOBALS.DJIA_SYMBOLS):
    df_ibm=pd.read_csv(f"{_GLOBALS.ROOT_PATH}Stocks/csv_{_GLOBALS.NUM_YRS}yrs/{symb}.csv", parse_dates=True, index_col=0)
    if (len(df_ibm)): # Check in case Yahoo failed to provide the data
        df_ibm=df1.join(df_ibm)
        df_ibm=df_ibm[['Close']]
        df_ibm=df_ibm.fillna(method='ffill')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1,1))

        x_train, y_train, x_test, y_test = load_data(df_ibm, look_back)
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_train = torch.from_numpy(y_train).type(torch.Tensor)
        y_test = torch.from_numpy(y_test).type(torch.Tensor)

        model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        loss_fn = torch.nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

        print(f'Training for... #{i+1} {symb}')
        hist = np.zeros(num_epochs)
        for t in range(num_epochs):
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            #model.hidden = model.init_hidden()

            # Forward pass
            model.to(device)
            x_train=x_train.to(device)
            y_train=y_train.to(device)
            y_train_pred = model(x_train)

            loss = loss_fn(y_train_pred, y_train)
            if t % 10 == 0 and t !=0:
                print("Epoch ", t, "MSE: ", loss.item())
            hist[t] = loss.item()

            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            optimiser.step()

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        torch.save(model.state_dict(), f"{model_save_dir}/model_{symb}.pt")
        if (_GLOBALS.SHOW_PLOTS):
            plt.plot(hist, label=f"{symb }Training loss")
            plt.legend()
            plt.show()

        y_train_pred = scaler.inverse_transform(y_train_pred.cpu().detach().numpy())
        y_train = scaler.inverse_transform(y_train.cpu().detach().numpy())
        trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))

        rmsfile.write(f"{symb},{trainScore}\n")

rmsfile.close()
