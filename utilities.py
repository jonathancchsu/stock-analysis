import os
import yfinance as yf

class Utilities:
  def download_data(start_date, end_date, ticker):
    """
    Download stock data using the yfinance API
    Parameters:
      start_date: string 
        The start date of the data in YYYY-MM-DD format
      end_date: string 
        The end date of the data in YYYY-MM-DD format
      ticker: string 
        The ticker of the stock 
    """
    tickerData = yf.Ticker(ticker)

    tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

    csv_file_path = os.path.join('stock_data', f'{ticker}.csv')
    tickerDf.to_csv(csv_file_path)