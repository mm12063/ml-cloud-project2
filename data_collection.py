import pandas as pd
import pandas_datareader.data as web
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os

DJIA_SYMBOLS = ['AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CSCO', 'CVX', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'JNJ', 'KO', 'JPM', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PG', 'TRV', 'UNH', 'CRM', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'DOW']
DATA_SOURCE = 'yahoo'

NUM_YRS = 15

today = date.today()
yesterday = today - timedelta(days=1)
n_years_ago = yesterday - relativedelta(years=NUM_YRS)
start_date = n_years_ago
end_date = yesterday

for i, symb in enumerate(DJIA_SYMBOLS):
	print(f"#{i+1} Collecting data for...{symb}")
	stock_df = web.DataReader(symb, DATA_SOURCE, start_date, end_date)
	print(f"Number of rows... {len(stock_df.index)}")

	dir = f"./data/csv_{NUM_YRS}yrs/"
	if not os.path.exists(dir):
		os.makedirs(dir)
	file = f"{dir}{symb}.csv"
	stock_df.to_csv(file)
	print("Done")
print("Complete")
