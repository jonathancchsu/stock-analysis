# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stock import *

class FCN(Stock):
  """
  FNC description
  """

  def __init__(self, stocks, ko, ki, s=100, g=27, t=252):
    """
    Parameters
    ----------
    stocks: list of Stock objects
            list of simulations 
    ko: double 
        KO
    ki: double 
        KI
    s : double
        strike price
    g : int
        guaranteed days (27 trading days)
    t : int 
        tenor
    """

    self.stocks = stocks
    self.ko = ko
    self.ki = ki
    self.strike = s
    self.tenor = t
    self.guranteed_days = g

  def __repr__(self):
    stock_tickers = []
    for stock in self.stocks:
      stock_tickers.append(stock.ticker)
    info = (f'Stocks (stocks={stock_tickers}, '
            f'ki={self.ki:.2f}, ko={self.ko:.2f}, s={self.strike:.2f})')
    return info
  
  def set_tenor(self, t):
    self.tenor = t

  def set_KI(self, ki):
    self.ki = ki

  def set_KO(self, ko):
    self.ko = ko
  
  def plot_stocks(self):
    plt.figure(figsize=(10, 8))
    # Ensure labels are present for the legend
    for stock in self.stocks:
      # You can use stock.ticker or any other unique property of the stock
      plt.plot(stock.historical_prices['aggregated_daily_returns'], label=stock.ticker)
    
    plt.xlabel('Date')
    plt.ylabel('Percent Aggregate Return') 
    plt.title(f'Historical Aggregate Return')
    # from {self.stocks[0].start_date} to {self.stocks[0].stock.end_date}
    plt.legend()
    plt.show()

  def backtest_KO(self, start_date, end_date):
    trading_dates = self.stocks[0].historical_prices['Date']
    if start_date and end_date not in trading_dates:
      msg = f'{start_date} and {end_date} are not trading dates.' 
    if start_date or end_date not in trading_dates:
      msg = f'{start_date} is not a trading date.' if start_date not in trading_dates \
            else f'{end_date} is not a trading date.'
      return msg
    # trading_dates = pd.to_datetime(self.stocks[0].historical_prices['Date'])
    # adjusted_start, adjusted_end = self.adjust_dates(start_date, end_date, trading_dates)
    ko_dates = []
    
    for date in trading_dates:
      if date < pd.to_datetime(start_date) or date > pd.to_datetime(end_date):
        continue
      
      try:
        future_date = self.add_trading_days(date, trading_dates)
      except KeyError as e:
        print(f"Date not found in trading dates: {e}")
        continue 
      
      date_mask = (trading_dates >= date) & (trading_dates <= future_date)
      valid_dates = trading_dates[date_mask]

      all_stocks_above_ko = True
      # Check each trading day from start date to future date
      for single_date in valid_dates:
        for stock in self.stocks:
          # Fetch the daily returns for this single_date
          stock_daily_data = stock.historical_prices.loc[
            pd.to_datetime(stock.historical_prices['date']) == single_date]

          # Check if the returns are above the ko threshold
          if not (stock_daily_data['adjusted_daily_returns'] > self.ko).any():
            all_stocks_above_ko = False
            break
        
        if not all_stocks_above_ko:
            break
        
      # If all stocks are above ko on this particular day, add the start date to ko_dates
      if all_stocks_above_ko:
        ko_dates.add(date.strftime('%Y-%m-%d'))

    return ko_dates

  def graph_backtest_KO(self, start_date, end_date, ko_dates):
    plt.figure(figsize=(10, 8))

    # Convert string dates to datetime objects for plotting
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Plot each stock's historical returns
    for stock in self.stocks:
      dates = pd.to_datetime(stock.historical_prices['date'])  # Ensure dates are in datetime format
      returns = stock.historical_prices['adjusted_daily_returns']
      plt.plot(dates, returns, label=stock.ticker)
    
    # Highlight the KO periods
    for ko_date in ko_dates:
      ko_start = pd.to_datetime(ko_date)
      ko_end = ko_start + pd.Timedelta(days=self.tenor)
      plt.axvspan(ko_start, ko_end, color='lightcoral', alpha=0.3, label='KO Period' if ko_date == ko_dates[0] else "")

    # Add a horizontal line for the KO level
    plt.axhline(y=self.ko, color='black', linestyle='--', linewidth=2, label='KO Level')

    # Add vertical lines for start and end dates
    plt.axvline(x=start_dt, color='green', linestyle='--', linewidth=2, label='Start Date')
    plt.axvline(x=end_dt, color='blue', linestyle='--', linewidth=2, label='End Date')

    # Formatting the plot
    plt.xlabel('Date')
    plt.ylabel('Percent Aggregate Return')
    plt.title('Historical Aggregate Return with KO Highlights')
    plt.legend()

    # Improve date formatting on x-axis
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation

    plt.show()

  def backtest_KI(self, start_date, end_date):
    pass

  def backtest_exercise(self, start_date, end_date):
    pass

  def add_trading_days(self, start_date, trading_dates):
    """
    Adjust the start date to the nearest available trading day if it's not a trading day,
    then find the date that is 'tenor' trading days from the adjusted start date.

    Parameters:
    start_date (str or pd.Timestamp): Initial date.
    tenor (int): Number of trading days to move forward.
    trading_dates (pd.Series or array-like): Series or array of all trading dates.

    Returns:
    pd.Timestamp: The date that is 'tenor' trading days from the adjusted start date.
    """
    # Ensure trading_dates is a DatetimeIndex for proper functionality
    if not isinstance(trading_dates, pd.DatetimeIndex):
      trading_dates = pd.DatetimeIndex(trading_dates)
    
    start_date = pd.to_datetime(start_date)

    # Ensure start_date is a trading day or adjust to the next trading day
    if start_date not in trading_dates:
      # Find the nearest future trading day if start_date is not found
      future_dates = trading_dates[trading_dates >= start_date]
      if not future_dates.empty:
        start_date = future_dates[0]
      else:
        raise ValueError("No valid trading days available after the start date.")

    # Find the index of the start date
    start_idx = trading_dates.get_loc(start_date)

    # Calculate the index for the date that is 'tenor' days ahead, ensuring bounds are respected
    future_idx = min(start_idx + self.tenor, len(trading_dates) - 1)

    return self.stocks[0].historical_prices['date'][future_idx]
  
  def adjust_dates(self, start_date, end_date, trading_dates):
    """
    Adjust the start and end dates to the nearest trading days. If the start_date is not a trading day,
    adjust to the next available trading day. If the end_date is not a trading day, adjust to the 
    previous available trading day.

    Parameters:
    start_date (str or pd.Timestamp): The proposed initial date.
    end_date (str or pd.Timestamp): The proposed ending date.
    trading_dates (pd.Series or list): Series or list of all trading dates.

    Returns:
    tuple: Adjusted (start_date, end_in_date) as pd.Timestamps.
    """
    # Convert all dates to pandas datetime for consistency
    trading_dates = pd.to_datetime(trading_dates)
    start_date = pd.to_datetime(start_date) 
    end_date = pd.to_datetime(end_date)

    # Adjust start_date to the next available trading day if necessary
    if start_date not in trading_dates:
      future_dates = trading_dates[trading_dates >= start_date]
      if not future_dates.empty:
        start_date = future_dates[0]
      else:
        raise ValueError("No future trading days available after the proposed start date.")

    # Adjust end_date to the previous available trading day if necessary
    if end_date not in trading_dates:
      past_dates = trading_dates[trading_dates <= end_date]
      if not past_dates.empty:
        end_date = past_dates[-1]
      else:
        raise ValueError("No past trading days available before the proposed end date.")

    return start_date, end_date