import kfp
from kfp import dsl
import kfp.components as components


def step_get_latest_data(ticker: str):
    print("Getting data from Yahoo")
    from datetime import date
    from datetime import timedelta
    from dateutil.relativedelta import relativedelta
    from minio import Minio
    import os
    import glob
    import yfinance as yf

    NUM_YRS = 10
    ROOT_PATH = '/tmp/'
    DIR = 'stock-data/'
    FULL_LOCAL_DIR = f"{ROOT_PATH}{DIR}"

    minio_client = Minio(
        "172.21.107.4:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"

    today = date.today()
    yesterday = today - timedelta(days=1)
    n_years_ago = yesterday - relativedelta(years=NUM_YRS)
    start_date = n_years_ago
    end_date = yesterday

    if not os.path.exists(FULL_LOCAL_DIR):
        os.makedirs(FULL_LOCAL_DIR)

    print(f"Collecting data for...{ticker}")
    stock_df = yf.download(ticker, start_date, end_date)
    print(f"Number of rows... {len(stock_df.index)}")

    file = f"{FULL_LOCAL_DIR}{ticker}.csv"
    print(f"Saving to... {file}")
    stock_df.to_csv(file)
    print("All data downloaded and saved!")

    print("Moving all files to Minio")
    for local_file in glob.glob(FULL_LOCAL_DIR + '/**'):
        remote_path = os.path.join(minio_bucket, DIR, local_file[len(FULL_LOCAL_DIR):])
        minio_client.fput_object(minio_bucket, remote_path, local_file)
    print("All moved to Minio remote dir")


def step_pipeline_training(no_epochs: int, learning_rate: float, ticker: str):
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
    from minio import Minio

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
        def __init__(self, device, input_dim, hidden_dim, num_layers, output_dim):
            super(LSTM, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.device = device

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)
            out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            out = self.fc(out[:, -1, :])
            return out

    minio_client = Minio(
        "172.21.107.4:9000",
        access_key="minio",
        secret_key="minio123",
        secure=False
    )
    minio_bucket = "mlpipeline"

    INPUT_DIM = 1
    HIDDEN_DIM = 32
    NUM_LAYERS = 5
    OUTPUT_DIM = 1
    NUM_YRS = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    today = date.today()
    yesterday = today - timedelta(days=1)
    n_years_ago = yesterday - relativedelta(years=NUM_YRS)
    start_date = n_years_ago
    end_date = yesterday
    dates = pd.date_range(start_date, end_date, freq='B')
    df1 = pd.DataFrame(index=dates)

    look_back = 5
    num_epochs = no_epochs

    ROOT_PATH = '/tmp/'
    DIR = 'stock-data/'
    FULL_LOCAL_DIR = f"{ROOT_PATH}{DIR}"
    FULL_LOCAL_FILE = f"{FULL_LOCAL_DIR}{ticker}.csv"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    remote_file = f"{minio_bucket}/{DIR}{ticker}.csv"
    print("CSV to retrieve from minio:", remote_file)

    csv_found = False
    try:
        minio_client.fget_object(minio_bucket, remote_file, FULL_LOCAL_FILE)
        csv_found = True
    except:
        print(f"No CSV file found for {ticker}")

    if (csv_found):
        LOCAL_RMSE_DIR = "/tmp/rmses/"
        if not os.path.exists(LOCAL_RMSE_DIR):
            os.makedirs(LOCAL_RMSE_DIR)
        rmse_file_name = f"{ticker}_RMSE.csv"
        FULL_LOCAL_RMSE_FILE = f"{LOCAL_RMSE_DIR}{rmse_file_name}"
        rmsfile = open(f"{FULL_LOCAL_RMSE_FILE}", "w")

        df_ibm = pd.read_csv(FULL_LOCAL_FILE, parse_dates=True, index_col=0)
        if (len(df_ibm) > num_epochs * 10):  # Check in case Yahoo failed to provide the data
            df_ibm = df1.join(df_ibm)
            df_ibm = df_ibm[['Close']]
            df_ibm = df_ibm.fillna(method='ffill')
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df_ibm['Close'] = scaler.fit_transform(df_ibm['Close'].values.reshape(-1, 1))

            x_train, y_train, x_test, y_test = load_data(df_ibm, look_back)
            x_train = torch.from_numpy(x_train).type(torch.Tensor)
            y_train = torch.from_numpy(y_train).type(torch.Tensor)

            model = LSTM(device, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM,
                         num_layers=NUM_LAYERS)
            loss_fn = torch.nn.MSELoss()
            optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

            print(f'Training for...{ticker}')
            hist = np.zeros(num_epochs)
            for t in range(num_epochs):
                # Forward pass
                model.to(device)
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                y_train_pred = model(x_train)

                loss = loss_fn(y_train_pred, y_train)
                if t % 10 == 0 and t != 0:
                    print("Epoch ", t, "MSE: ", loss.item())
                hist[t] = loss.item()

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            LOCAL_MODEL_DIR = "/tmp/models/"
            if not os.path.exists(LOCAL_MODEL_DIR):
                os.makedirs(LOCAL_MODEL_DIR)

            FILE_NAME = f"model_{ticker}.pt"
            FULL_LOCAL_MODEL_FILE = f"{LOCAL_MODEL_DIR}{FILE_NAME}"

            print("-- Storing the model")
            torch.save(model.state_dict(), FULL_LOCAL_MODEL_FILE)
            print("Model stored")

            print("-- Moving model to Minio")
            REMOTE_PATH = f"{minio_bucket}/models/{FILE_NAME}"
            minio_client.fput_object(minio_bucket, REMOTE_PATH, FULL_LOCAL_MODEL_FILE)
            print("Model move complete")

            y_train_pred = scaler.inverse_transform(y_train_pred.cpu().detach().numpy())
            y_train = scaler.inverse_transform(y_train.cpu().detach().numpy())
            trainScore = math.sqrt(mean_squared_error(y_train[:, 0], y_train_pred[:, 0]))

            print("-- Writing to RMSE file")
            rmsfile.write(f"{ticker},{trainScore}\n")
            rmsfile.close()
            print("Writing complete")

            print("-- Moving RMSE file to Minio")
            full_rmse_remote_path = f"{minio_bucket}/rmses/{rmse_file_name}"
            minio_client.fput_object(minio_bucket, full_rmse_remote_path, FULL_LOCAL_RMSE_FILE)
            print("RMSE file move complete")

        else:
            print(f"Couldn't train – not enough data in CSV. Check Yahoo in case it's down for {ticker}.")

        print("Training step complete!")
    else:
        print(f"No CSV for {ticker}")


def step_serve_model_to_inference(ticker: str):
    from kubernetes import client
    from kserve import KServeClient
    from kserve import constants
    from kserve import utils
    from kserve import V1beta1InferenceService
    from kserve import V1beta1InferenceServiceSpec
    from kserve import V1beta1PredictorSpec
    from kserve import V1beta1TFServingSpec
    from datetime import datetime

    namespace = utils.get_default_target_namespace()

    name = f'model_{ticker}'
    kserve_version = 'v1beta1'
    api_version = constants.KSERVE_GROUP + '/' + kserve_version

    isvc = V1beta1InferenceService(
        api_version=api_version,
        kind=constants.KSERVE_KIND,
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=namespace,
            annotations={'sidecar.istio.io/inject': 'false'}
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                service_account_name="sa-minio-kserve",
                pytorch=(V1beta1TFServingSpec(
                    storage_uri="s3://mlpipeline/models/"))))
    )

    KServe = KServeClient()
    KServe.create(isvc)


comp_get_latest_data = components.create_component_from_func(step_get_latest_data,
                                                             base_image="docker.io/mm12063/pand-dr-minio:2.0")
comp_pipeline_training = components.create_component_from_func(step_pipeline_training,
                                                               base_image="docker.io/mm12063/pand-dr-minio:3.0")


@dsl.pipeline(
    name='DIJA-stock-prediction-pipeline',
    description='Train models to predict stock prices of DJIA companies'
)
def dija_pipeline(no_epochs: int, learning_rate: float, ticker: str):
    step1 = step_get_latest_data(ticker)
    step2 = step_pipeline_training(no_epochs, learning_rate, ticker)
    step3 = step_serve_model_to_inference(ticker)
    step2.after(step1)
    step3.after(step2)


if __name__ == "__main__":
    client = kfp.Client()

    # Defaults args
    arguments = {
        "no_epochs": 120,
        "learning_rate": 0.01,
        "ticker": "IBM"
    }

    run_directly = 1

    if (run_directly == 1):
        client.create_run_from_pipeline_func(dija_pipeline, arguments=arguments,
                                             experiment_name="DIJA-stock-prediction-pipeline-exp")
    else:
        kfp.compiler.Compiler().compile(pipeline_func=dija_pipeline, package_path='DIJA-stock-prediction-pipeline.yaml')
        client.upload_pipeline_version(pipeline_package_path='DIJA-stock-prediction-pipeline.yaml',
                                       pipeline_version_name="1.0",
                                       pipeline_name="DIJA Stock Prediction pipeline", description="just for testing")