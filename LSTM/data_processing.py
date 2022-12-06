import pandas_datareader.data as web
from datetime import date
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import os
import yfinance as yf
import VARS as _GLOBALS

today = date.today()
yesterday = today - timedelta(days=1)
n_years_ago = yesterday - relativedelta(years=_GLOBALS.NUM_YRS)
start_date = n_years_ago
end_date = yesterday

for i, symb in enumerate(_GLOBALS.DJIA_SYMBOLS):
	print(f"#{i+1} Collecting data for...{symb}")
	stock_df = yf.download(symb, start_date, end_date)
	print(f"Number of rows... {len(stock_df.index)}")

	dir = f"{_GLOBALS.ROOT_PATH}Stocks/csv_{_GLOBALS.NUM_YRS}yrs/"
	if not os.path.exists(dir):
		os.makedirs(dir)
	file = f"{dir}{symb}.csv"
	stock_df.to_csv(file)
	print("Done")
print("Complete")
