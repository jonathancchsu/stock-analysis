import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from stock import *

class FCN:
  """
  The FNC (Fixed Coupon Notes) is designed to contain the functions used for analyzing 
  FNC. Including backtesting with past data and gain information based on simulated data.
  """

  def __init__(self, stocks, ko=120, ki=60, s=80, g=27, t=252, r=0.048):
    """
    Parameters
    ----------
    stocks: list of Stock objects
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
    r : double
        rate
    """

    self.stocks = stocks
    self.ko = ko
    self.ki = ki
    self.strike = s
    self.guaranteed_days = g
    self.tenor = t
    self.rate = r
    

  def __repr__(self):
    """Returns a formatted string containing the data members of the class"""
    stock_tickers = []
    for stock in self.stocks:
      stock_tickers.append(stock.ticker)
    info = (f'Stocks (stocks={stock_tickers}, '
            f'ki={self.ki:.2f}, ko={self.ko:.2f}, s={self.strike:.2f})')
    return info
  
  def set_tenor(self, t):
    """Sets the tenor to a different length"""
    self.tenor = t

  def set_KO(self, ko):
    """Sets the KO to a different value"""
    self.ko = ko

  def set_KI(self, ki):
    """Sets the KI to a different value"""
    self.ki = ki
  
  def plot_stocks(self):
    """
    Generates the plot of the aggregated historical returns of the basket of stocks
    given, displays the changes of returns in percentages
    """
    plt.figure(figsize=(10, 8))
    for stock in self.stocks:
      plt.plot(stock.historical_prices['aggregated_historical_returns'], label=stock.ticker)
    
    plt.xlabel('Date')
    plt.ylabel('Percent Aggregate Return') 
    plt.title(f'Aggregate Historical Return')
    plt.legend()
    plt.show()

  def check_dates(self, start_date, end_date):
    """
    Checks if the dates are contained within the dataset
    Parameters:
      start_date: string 
        start_date in YYYY-MM-DD format
      end_date: string 
        end_date in YYYY-MM-DD format
    """
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
      return False

  def backtest_fcn(self, start_date, end_date):
    """
    Backtest using historical data to see which dates will KO, returns
    in a list of datetime Objects
    Parameters:
      start_date: string 
        start_date in YYYY-MM-DD format
      end_date: string 
        end_date in YYYY-MM-DD format
    Returns:
      ko_dates: list[datetime Objects]
        list of datetime Objects in YYYY-MM-DD
    """
    # Converting the string to datetime
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()
    trading_dates = self.stocks[0].historical_prices['Date']
    check_dates = self.check_dates(start_date, end_date)
    
    if not check_dates:
      pass 
    else:
      print(check_dates)
    
    start_idx = trading_dates[trading_dates == start_date].index[0]
    # Adjust the start date to avoid the guaranteed period
    adjusted_start_date = trading_dates.iloc[start_idx + self.guaranteed_days]
    current_date = adjusted_start_date
    ko_dates = []
    ki_dates = []
    exercise_dates = []

    # iterate through tenor amount of trading days to check for KO 
    while current_date <= end_date:
      current_date_idx = trading_dates[trading_dates == current_date].index[0]

      date_after_tenor = trading_dates.iloc[current_date_idx + self.tenor]
      ko_check = True
      ki_check = True
      exercise_check = False

      # check if all 3 stock beat KO within the same time period 
      for stock in self.stocks:
        stock_prices = stock.calculate_aggregate_returns(current_date, date_after_tenor)
        
        if stock_prices.max() < self.ko:
          ko_check = False
          break

        if stock_prices.min() > self.ki:
          ki_check = False
      
      # append if all stocks beat KO
      if ko_check:
        ko_dates.append(current_date.strftime('%Y-%m-%d'))
      
      if ki_check:
        ki_dates.append(current_date.strftime('%Y-%m-%d'))
        for stock in self.stocks:
          if stock_prices[-1] < self.strike:
            exercise_check = True
            break

        if exercise_check:
          exercise_dates.append(current_date.strftime('%Y-%m-%d'))

      # continue iterating
      current_date = trading_dates.iloc[current_date_idx + 1]

    return {
      'KO Dates': ko_dates,
      'KI Dates': ki_dates,
      'Exercise Dates': exercise_dates
    }

  def graph_backtest_fcn(self, start_date, end_date):
    """
    Graphs the backtest_KO results given the start and end dates and the list 
    of dates that will KO
    Parameters:
      start_date: string 
        start_date in YYYY-MM-DD format
      end_date: string 
        end_date in YYYY-MM-DD format
      ko_dates: list[datetime Objects]
        list of datetime Objects in YYYY-MM-DD
    """
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

    plt.axhline(y=self.ko, color='red', linestyle='--', linewidth=2, label='KO Level')
    plt.axhline(y=self.ki, color='green', linestyle='--', linewidth=2, label='KI Level')
    plt.axhline(y=self.strike, color='black', linestyle='--', linewidth=2, label='Strike Level')

    # Add vertical lines for start and end dates
    plt.axvline(x=start_dt, color='blue', linestyle='--', linewidth=2, label='Start Date')
    plt.axvline(x=end_dt, color='blue', linestyle='--', linewidth=2, label='End Date')

    # Formatting the plot
    plt.xlabel('Date')
    plt.ylabel('Percent Aggregate Return')
    plt.title('Historical Aggregate Return with FCN Conditions')
    plt.legend()

    # Improve date formatting on x-axis
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()  # Rotation

    plt.show()

  # def simulate_KO(self, list_of_simulations, num_trials=1000, start_days_from_today=0):
  #   """
  #   Test if stock(s) will KO given the simulations of the stocks and the number of
  #   trials that would be conducted. Can also customize the testing period by giving 
  #   the amount of days the test starts from current date. Graphs the result and returns
  #   a list of percentages of KO based on the given parameters.
  #   Parameters:
  #     list_of_simulations: list[np.array]
  #       list of simulated values of the stock(s)
  #     num_trials: int
  #       number of times the simulation will be conducted
  #     start_days_from_today: int
  #       numbers of days away from today to start the testing period
  #   Returns:
  #     list_of_ko_percentage: list[string]
  #       a list of string that contains the KO percentages of the stock(s) simulated
  #   """
  #   start_idx = start_days_from_today + self.guaranteed_days
  #   end_idx = self.tenor
  #   list_of_ko_percentage = []

  #   for simulation in list_of_simulations:
  #     list_of_prices = simulation.generate_simulated_stock_values(num_trials)
  #     count = 0.0

  #     # Initialize the plot for each stock simulation
  #     plt.figure(figsize=(10, 6))
      
  #     plt.xlabel('Days')
  #     plt.ylabel('Stock Price')
  #     plt.axhline(y=self.ko, color='black', linestyle='--', label='KO Threshold', zorder=3)

  #     for prices in list_of_prices:
  #       modified_prices = prices[start_idx:end_idx]
  #       days = np.arange(len(prices))

  #       # Check if any price in the modified range exceeds the KO threshold
  #       if np.max(modified_prices) > self.ko:
  #         count += 1
  #         plt.plot(days, prices, label='Above KO', color='red', alpha=0.5)
  #       else:
  #         plt.plot(days, prices, label='Below KO', color='green', alpha=0.5)

  #     list_of_ko_percentage.append(f'{simulation.ticker}: {count / num_trials * 100:.2f}% KO')
  #     plt.title(f'{num_trials} Simulation Results for {simulation.ticker}: {count / num_trials * 100:.2f}% KO')

  #     # Highlight the test period
  #     plt.axvspan(start_days_from_today, end_idx, color='lightcoral', alpha=0.3, label='Test Period')

  #     # Add legend and show plot
  #     handles, labels = plt.gca().get_legend_handles_labels()
  #     by_label = dict(zip(labels, handles))  # Remove duplicate labels
  #     plt.legend(by_label.values(), by_label.keys())
  #     plt.show()

  #   return list_of_ko_percentage
  
  def simulate_fcn(self, num_trials=1000, start_days_from_today=0, time_steps=500):
    """
    Simulate and test for KO, KI, and Exercise conditions based on the generated correlated stock prices.
    Parameters:
      num_trials (int): Number of simulations to run.
      start_days_from_today (int): Offset from today to start the testing period.

    Returns:
      dict: Contains percentages of KO, KI, and Exercise simulations.
    """
    # Generate random indices for the trials to plot
    plot_indices = np.random.choice(num_trials, 5, replace=False)
    start_idx = start_days_from_today
    end_idx = start_days_from_today + self.tenor
    results = {'KO': [], 'KI': [], 'Exercise': []}

    for trial in range(num_trials):
      # Generate the correlated random walks
      simulated_prices = self.generate_correlated_random_walks(time_steps)
      
      ko_flag, ki_flag, exercise_flag = False, True, False
      
      # Iterate through each stock's simulation
      for idx, stock in enumerate(self.stocks):
        prices = simulated_prices[idx, :]
        days = np.arange(time_steps)

        # Check KO and KI conditions
        if np.any(prices[start_idx:end_idx] > self.ko):
          ko_flag = True
        if not np.any(prices[start_idx:end_idx] < self.ki):
          ki_flag = False
        
        # Check exercise condition
        if prices[end_idx - 1] < self.strike:
          exercise_flag = True

      # Record the results for each trial
      if ko_flag:
        results['KO'].append(1)
      else:
        results['KO'].append(0)
      
      if ki_flag:
        results['KI'].append(1)
        if exercise_flag:
          results['Exercise'].append(1)
        else:
          results['Exercise'].append(0)
      else:
        results['KI'].append(0)
        results['Exercise'].append(0)

      # Plot for selected trials
      if trial in plot_indices:
        plt.figure(figsize=(15, 8))
        for idx, stock in enumerate(self.stocks):
            plt.plot(days, simulated_prices[idx, :], label=f'{stock.ticker} Walk', alpha=0.5)
        plt.axvspan(start_days_from_today, end_idx, color='lightcoral', alpha=0.2, label='Test Period')
        plt.axhline(y=self.ko, color='red', linestyle='--', label='KO Threshold')
        plt.axhline(y=self.ki, color='green', linestyle='--', label='KI Threshold')
        plt.axhline(y=self.strike, color='blue', linestyle='--', label='Strike Price')
        plt.xlabel('Days')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Simulation for Trial {trial + 1}')
        plt.legend()
        plt.show()

    # Calculate percentages
    ko_percentage = 100 * np.mean(results['KO'])
    ki_percentage = 100 * np.mean(results['KI'])
    exercise_percentage = 100 * np.mean(results['Exercise'])
    
    print({
      'KO Percentage': ko_percentage,
      'KI Percentage': ki_percentage,
      'Exercise Percentage': exercise_percentage
    })
    return {
      'KO Percentage': ko_percentage,
      'KI Percentage': ki_percentage,
      'Exercise Percentage': exercise_percentage
    }

  def calculate_correlation(self, days):
    prices = pd.concat([stock.historical_prices['Close'].iloc[-days:] for stock in self.stocks], axis=1)
    prices.columns = [stock.ticker for stock in self.stocks]
    returns = prices.pct_change(fill_method=None)
    correlation_matrix = returns.corr()
    return correlation_matrix.to_numpy()
  
  def generate_correlated_random_walks(self, time_steps):
    vols = [stock.sigma for stock in self.stocks]
    basket_size = len(self.stocks)
    correlation_matrix = self.calculate_correlation(self.stocks[0].train)
    Q = np.linalg.cholesky(correlation_matrix)
    stock_prices = np.full((basket_size, time_steps), 100.0)

    Q_transpose = Q.T.conj()
    
    if not np.allclose(correlation_matrix, np.dot(Q, Q_transpose)):
      print(f'correlation_matrix: {correlation_matrix}')
      print(f'ckeck: {np.dot(Q, Q_transpose)}')

    for t in range(1, time_steps):
      random_array = np.random.standard_normal(basket_size)
      epsilon_array = np.inner(random_array, Q)

      for n in range(basket_size):
        dt = 1 / time_steps
        S = stock_prices[n, t-1]
        sigma = vols[n]
        epsilon = epsilon_array[n]

        stock_prices[n, t] = S * np.exp((self.rate - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * epsilon)
        
    return stock_prices
  
  def graph_random_walks(self, stock_prices, time_steps):
    basket_size = len(self.stocks)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    days = [t for t in range(time_steps)]
    stock_ticker = [stock.ticker for stock in self.stocks]
    
    for n in range(basket_size):
      ax.plot(days, stock_prices[n], label = '{}'.format(stock_ticker[n]))

    plt.grid()
    plt.xlabel('Days')
    plt.ylabel('Stock Prices')
    plt.legend()
    plt.show()

  def print_dates(self, dates_dict):
    # Output DataFrame for all results
    all_results_df = pd.DataFrame()

    for category, dates in dates_dict.items():
      # Convert trading dates to datetime and create Series
      trading_dates = pd.to_datetime(self.stocks[0].historical_prices['Date'])
      dates_series = pd.to_datetime(pd.Series(dates))

      # Only consider dates that are in the trading dates
      valid_dates_series = dates_series[dates_series.isin(trading_dates)]

      # Function to find groups of consecutive trading dates
      def find_consecutive_dates(input_dates):
        # Calculate the index positions of input_dates within trading_dates
        input_date_indices = trading_dates.searchsorted(input_dates)
        # Find gaps by comparing consecutive indices - consecutive trading dates have a difference of 1
        gaps = pd.Series(input_date_indices).diff() - 1
        # Group by cumsum of gaps; consecutive days have a cumsum that increases when a gap occurs
        groups = (gaps > 0).cumsum()
        return input_dates.groupby(groups)

      grouped_dates = find_consecutive_dates(valid_dates_series)

      grouped_dates_summary = {}
      for i, (key, group) in enumerate(grouped_dates):
        if not group.empty:
          first_date = group.iloc[0].strftime('%Y-%m-%d')
          last_date = group.iloc[-1].strftime('%Y-%m-%d')
          grouped_dates_summary[i + 1] = {'First Date': first_date, 'Last Date': last_date}

      # Create a DataFrame for current category results
      df_grouped_dates = pd.DataFrame.from_dict(grouped_dates_summary, orient='index')
      df_grouped_dates['Category'] = category  # Add a column for the category

      # Append to the all results DataFrame
      all_results_df = pd.concat([all_results_df, df_grouped_dates])

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    print(all_results_df)
    return all_results_df

