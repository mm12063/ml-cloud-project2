import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import torch.nn as nn
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas_datareader.data as web
import yfinance as yf
import glob

ROOT_PATH = './'
DJIA_SYMBOLS = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW']
NUM_YRS = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def store_data():
    DJIA_SYMBOLS = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO',
                    'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT',
                    'DIS', 'DOW']
    NUM_YRS = 10
    today = date.today()
    yesterday = today - timedelta(days=1)
    n_years_ago = yesterday - relativedelta(years=NUM_YRS)
    start_date = n_years_ago
    end_date = yesterday

    dir = f"{ROOT_PATH}Stocks/csv_{NUM_YRS}yrs/"
    for local_file in glob.glob(dir + '/**'):
        print(local_file)


    exit(0)

    for i, symb in enumerate(DJIA_SYMBOLS):
        print(f"#{i + 1} Collecting data for...{symb}")
        stock_df = yf.download(symb, start_date, end_date)
        print(f"Number of rows... {len(stock_df.index)}")

        dir = f"{ROOT_PATH}Stocks/csv_{NUM_YRS}yrs/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        file = f"{dir}{symb}.csv"
        stock_df.to_csv(file)
        print("Done")
    print("Complete")





def load_data(stock, look_back):
    data_raw = stock.values
    data = []
    
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


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


def main():
    input_dim = 1
    hidden_dim = 32
    num_layers = 5
    output_dim = 1

    today = date.today()
    yesterday = today - timedelta(days=1)
    n_years_ago = yesterday - relativedelta(years=NUM_YRS)
    start_date = n_years_ago
    end_date = yesterday
    dates = pd.date_range(start_date, end_date, freq='B')
    df1 = pd.DataFrame(index=dates)

    store_data(start_date, end_date)

    look_back = 5
    num_epochs = 100

    # Create the dir
    model_save_dir = f'{ROOT_PATH}Model/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    #Remove any existing RMSE file
    try:
      os.remove(f"{model_save_dir}RMSE.csv")
    except OSError:
      pass

    rmsfile = open(f"{model_save_dir}RMSE.csv", "w")
    for i, symb in enumerate(DJIA_SYMBOLS):
        df_ibm=pd.read_csv(f"{ROOT_PATH}Stocks/csv_{NUM_YRS}yrs/{symb}.csv", parse_dates=True, index_col=0)
        if (len(df_ibm) > num_epochs * 10): # Check in case Yahoo failed to provide the data
            df_ibm=df1.join(df_ibm)
            df_ibm=df_ibm[['Close']]
            df_ibm=df_ibm.fillna(method='ffill')
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1,1))

            x_train, y_train, x_test, y_test = load_data(df_ibm, look_back)
            x_train = torch.from_numpy(x_train).type(torch.Tensor)
            y_train = torch.from_numpy(y_train).type(torch.Tensor)

            model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
            loss_fn = torch.nn.MSELoss()
            optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

            print(f'Training for... #{i+1} {symb}')
            hist = np.zeros(num_epochs)
            for t in range(num_epochs):
                # Forward pass
                model.to(device)
                x_train=x_train.to(device)
                y_train=y_train.to(device)
                y_train_pred = model(x_train)

                loss = loss_fn(y_train_pred, y_train)
                if t % 10 == 0 and t !=0:
                    print("Epoch ", t, "MSE: ", loss.item())
                hist[t] = loss.item()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            torch.save(model.state_dict(), f"{model_save_dir}/model_{symb}.pt")

            y_train_pred = scaler.inverse_transform(y_train_pred.cpu().detach().numpy())
            y_train = scaler.inverse_transform(y_train.cpu().detach().numpy())
            trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))

            rmsfile.write(f"{symb},{trainScore}\n")

    rmsfile.close()

if __name__ == '__main__':
    store_data()
    # main()