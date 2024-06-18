import os
import yfinance as yf

class Utilities:
  def download_data(start_date, end_date, ticker):
    tickerData = yf.Ticker(ticker)

    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

    csv_file_path = os.path.join('stock_data', f'{ticker}.csv')
    tickerDf.to_csv(csv_file_path)