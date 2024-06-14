3# -*- coding: utf-8 -*-
import requests
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
# from datetime import datetime

class Stock:
    """
    The Stock class is designed to contain the information of a Stock given the ticker
    and the given time frame using Polygon.api
    """

    def __init__(self, t, s, e):
        """
        Parameters
        ----------
        t : string
            The ticker of the stock, e.g. 'NVDA'
        s : string
            The start date of observing the closing price e.g. '2021-06-30'
        e : string
            The end date of observing the closing price e.g. '2021-06-30'
        
        note: Polygon.io supplies only up to 2 years of stock data from current day with the free API

        Initialized Variables (eager loading)
        ----------
        historical_prices_USD: np array
            The historical prices of the stock in USD during the given time frame
        historical_prices: np array
            The historical prices of the stock with changes in percentages and 100 as baseline starting point
            during the given time frame
        avg_daily_return: float
            The historical returns of the stock during the given time frame
        avg_daily_return_sigma: float
            The standard deviation of the historical returns of the stock during the given time frame
        """
        if self.validate_dates(s, e):
            self.ticker = t
            self.start_date = s
            self.end_date = e
            self.historical_prices_USD = self.fetch_historical_prices()
            self.historical_prices = self.historical_aggre_return()
            self.avg_daily_return, self.avg_daily_return_sigma = self.calculate_mu_sigma()
            # Store each parameter as a data member and initialize the variables for the stock class
        else: 
            raise ValueError("Provided dates do not meet the required conditions (within 2 years from current date).")
        
    def __repr__(self):
        """Returns a formatted string containing the data members of the class"""

        info = (f'Stock (t=${self.ticker}, s={self.start_date}, e={self.end_date}, '
                f'avg_daily_return={self.avg_daily_return:.2f}, avg_daily_return_signma={self.avg_daily_return_sigma:.2f} '
                f'historical_prices={self.historical_prices} ')
        return info
    
    def get_dates(self):
        """Returns a formatted string containing the ticker and dates of a Stock"""
        info = (f'Stock (t=${self.ticker}, s={self.start_date}, e={self.end_date}')
        return info

    def set_dates(self, s, e):
        """
        Resets the date of the stock, re-fetches the historical prices, and 
        recalculates the avg_daily_return and avg_daily_return_sigma
        """
        self.start_date = s
        self.end_date = e
        self.historical_prices_USD = self.fetch_historical_prices()
        self.historical_prices = self.historical_aggre_return()
        self.avg_daily_return, self.avg_daily_return_sigma = self.calculate_mu_sigma(self.historical_prices)

    def plot_historical_prices(self):
        plt.figure(figsize = (10,8))
        plt.plot(self.historical_prices)
        plt.xlabel('Date')
        plt.ylabel('Percent Aggregate Return') 
        plt.title('Historical Aggregate Return')
        plt.show()
    
    def fetch_historical_prices(self):
        """
        Fetches the stock data for historical prices using the Polygon.io API

        Returns:
        historical_prices_USD: np array 
            The closed prices during the chosen time period 
        """
        # load_dotenv()
        # api_key = os.getenv('API_KEY')
        api_key = 'yNTTUyB6GMUYWZFOn6_xTjc18U4jGLj3'
        
        # print(f'current day is: {current_date}, date_two_years_ago is: {date_two_years_ago}')
        current_date, date_two_years_ago = self.fetch_date_range()
        url = f'https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/day/{date_two_years_ago}/{current_date}?adjusted=true&sort=asc&apiKey={api_key}'
        
        response = requests.get(url)
        data = response.json()
        if 'results' in data:
            df = pd.DataFrame(data['results'])
            # print(data)
            historical_prices_USD = df['c']
            # Convert the list to a numpy array
            if not isinstance(historical_prices_USD, np.ndarray):
                historical_prices_USD = np.array(historical_prices_USD)
            return historical_prices_USD
        else:
            print("No data available for the specified dates and ticker.")
    
    def calculate_mu_sigma(self):
        """
        Calculate the daily average return and the standard deviation from an array of closed stock prices.

        Returns:
        average_daily_return: float
            The daily average return as a percentage.
        avg_daily_return_sigma: float 
            The standard deviation of the daily returns.
        """
        # Calculate daily returns
        # Using the formula: (P_t - P_t-1) / P_t-1, where P_t is the price at time t
        daily_returns = (self.historical_prices_USD[1:] - self.historical_prices_USD[:-1]) / self.historical_prices_USD[:-1]
        # Calculate the average of these daily returns
        average_return = np.mean(daily_returns)

        return average_return, daily_returns.std()
    
    def historical_aggre_return(self):
        """
        Calculate the aggregated daily average returns from an array of closed stock prices.

        Returns:
        historical_prices: np array
            A list of aggregated daily average returns with 100 as baseline starting point
        """
        # Calculate aggregated daily returns
        # Using the formula: (P_t - P_0) / P_0, where P_t is the price at time t
        # print(self.historical_prices_USD)
        daily_returns = (self.historical_prices_USD[1:] - self.historical_prices_USD[0]) / self.historical_prices_USD[0]
        # Adding 0 at the start for baseline for day 1 of observation
        adjusted_daily_returns = np.insert(daily_returns, 0, 0)
        # Making the list into percentage changes
        adjusted_daily_returns = (adjusted_daily_returns + 1) * 100
        current_date, date_two_years_ago = self.fetch_date_range()
        trading_days = self.get_trading_days(date_two_years_ago, current_date)
        trading_days_dt = [datetime.datetime.strptime(date, '%Y-%m-%d') for date in trading_days]
        df_trading_days = pd.DataFrame({'date': trading_days_dt})
        df_adjusted_daily_returns = pd.DataFrame({'adjusted_daily_returns': adjusted_daily_returns})
        df = pd.concat([df_trading_days, df_adjusted_daily_returns], axis=1)
        # print(df)
        return df
    
    def validate_dates(self, s, e):
        """
        Validates that the start date is within the past 2 years from today
        and the end session. The function checks:
        1. The start date is no more than 2 years in the past from today.
        2. The end date is not before today.
        """
        try:
            start_date = datetime.datetime.strptime(s, "%Y-%m-%d").date()
            end_date = datetime.datetime.strptime(e, "%Y-%m-%d").date()
        except ValueError:
            print("Error: Incorrect date format. Please use YYYY-MM-DD.")
            return False

        # Get today's date
        today = datetime.date.today()
        # print(today)

        # Calculate two years ago from today
        two_years_ago = today.replace(year=today.year - 2)
        # print(two_years_ago)

        # Check if the start date is within the last two years
        if start_date < two_years_ago or start_date > today:
            print(f"The start date {start_date} must be within the last two years from today.")
            return False

        # Check if the end date is not after today
        if end_date > today:
            print(f"The end date {end_date} must not be after today.")
            return False

        # If both conditions are met
        return True
    
    def get_trading_days(self, start_date, end_date, market='NYSE'):
        """
        Returns a list of trading days between start_date and end_date for the given market.
        
        Parameters:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        market (str): Market identifier (e.g., 'NYSE' for New York Stock Exchange).
        
        Returns:
        list: List of trading days as strings in 'YYYY-MM-DD' format.
        """
        # Get the market calendar
        cal = mcal.get_calendar(market)
        
        # Fetch trading days
        trading_days = cal.schedule(start_date=start_date, end_date=end_date)
        return trading_days.index.strftime('%Y-%m-%d').tolist()
    
    def fetch_date_range(self):
        current_date_time = datetime.datetime.now()
        current_date = current_date_time.date()
        try:
            # Attempt to create a new date two years in the past
            date_two_years_ago = datetime.date(current_date.year - 2, current_date.month, current_date.day)
        except ValueError:
            # This block executes if the current day is Feb 29 and the year two years ago is not a leap year
            # Adjust to Feb 28 of the year two years ago
            date_two_years_ago = datetime.date(current_date.year - 2, current_date.month, 28)

        return current_date, date_two_years_ago