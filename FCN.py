import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stock import *

class FCN:
  """
  FNC description
  """

  def __init__(self, stocks, ko=120, ki=60, s=80, g=27, t=252):
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
    for stock in self.stocks:
      plt.plot(stock.historical_prices['aggregated_historical_returns'], label=stock.ticker)
    
    plt.xlabel('Date')
    plt.ylabel('Percent Aggregate Return') 
    plt.title(f'Aggregate Historical Return')
    plt.legend()
    plt.show()

  def backtest_KO(self, start_date, end_date):
    # Converting the string to datetime
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    print(f'start_date: {start_date}, end_date: {end_date}')
    trading_dates = self.stocks[0].historical_prices['Date']
    
    msg = ""
    # Check if the start and end dates are in the data
    if not trading_dates.isin([start_date]).any():
      msg += f'{start_date} is not a trading date. '
    if not trading_dates.isin([end_date]).any():
      msg += f'{end_date} is not a trading date. '

    if msg:
      return msg
    else:
      pass 
    
    start_idx = trading_dates[trading_dates == start_date].index[0]
    # Adjust the start date to avoid the guaranteed period
    adjusted_start_date = trading_dates.iloc[start_idx + self.guranteed_days]
    current_date = adjusted_start_date
    ko_dates = []

    while current_date <= end_date:
      # print(current_date)
      current_date_idx = trading_dates[trading_dates == current_date].index[0]

      date_after_tenor = trading_dates.iloc[current_date_idx + self.tenor]
      all_stocks_ko = True

      for stock in self.stocks:
        stock_prices = stock.calculate_aggre_returns(current_date, date_after_tenor)
        if stock_prices.max() < self.ko:
          all_stocks_ko = False
          break
      
      if all_stocks_ko:
        date_str = current_date.strftime('%Y-%m-%d')
        ko_dates.append(date_str)

      current_date = trading_dates.iloc[current_date_idx + 1]

    return ko_dates

  def graph_backtest_KO(self, start_date, end_date, ko_dates):
    plt.figure(figsize=(10, 8))

    # Convert string dates to datetime objects directly for plotting
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Ensure date formatting for plotting; plot each stock's historical returns
    for stock in self.stocks:
      # Ensure the Date column is in datetime format for plotting
      if 'Date' in stock.historical_prices:
        stock.historical_prices['Date'] = pd.to_datetime(stock.historical_prices['Date'])
        plt.plot(stock.historical_prices['Date'], stock.historical_prices['aggregated_historical_returns'], label=stock.ticker)
      else:
        # Assuming date is the index
        plt.plot(stock.historical_prices.index, stock.historical_prices['aggregated_historical_returns'], label=stock.ticker)

    # Highlight the KO periods
    for ko_date in ko_dates:
      ko_start = pd.to_datetime(ko_date)
      ko_end = ko_start + pd.Timedelta(days=self.tenor)
      plt.axvspan(ko_start, ko_end, color='lightcoral', alpha=0.5, label='KO Period' if ko_date == ko_dates[0] else "")

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

  def simulate_KO(self, list_of_simulations, num_trials=1000, start_days_from_today=0):
    start_idx = start_days_from_today + self.guranteed_days
    end_idx = self.tenor
    list_of_ko_percentage = []

    for simulation in list_of_simulations:
      list_of_prices = simulation.generate_simulated_stock_values(num_trials)
      count = 0.0

      # Initialize the plot for each stock simulation
      plt.figure(figsize=(10, 6))
      
      plt.xlabel('Days')
      plt.ylabel('Stock Price')
      plt.axhline(y=self.ko, color='black', linestyle='--', label='KO Threshold')

      for prices in list_of_prices:
        modified_prices = prices[start_idx:end_idx]
        days = np.arange(len(prices))

        # Check if any price in the modified range exceeds the KO threshold
        if np.max(modified_prices) > self.ko:
          count += 1
          plt.plot(days, prices, label='Above KO', color='red', alpha=0.5)
        else:
          plt.plot(days, prices, label='Below KO', color='green', alpha=0.5)

      list_of_ko_percentage.append(f'{simulation.ticker}: {count / num_trials * 100:.2f}% KO')
      plt.title(f'{num_trials} Simulation Results for {simulation.ticker}: {count / num_trials * 100:.2f}% KO')

      # Highlight the test period
      plt.axvspan(start_days_from_today, end_idx, color='lightcoral', alpha=0.3, label='Test Period')

      # Add legend and show plot
      handles, labels = plt.gca().get_legend_handles_labels()
      by_label = dict(zip(labels, handles))  # Remove duplicate labels
      plt.legend(by_label.values(), by_label.keys())
      plt.show()

    return list_of_ko_percentage

  def backtest_KI(self, start_date, end_date):
    pass

  def backtest_exercise(self, start_date, end_date):
    pass
