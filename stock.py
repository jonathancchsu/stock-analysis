# -*- coding: utf-8 -*-
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from dotenv import load_dotenv
# import os


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
        
        self.ticker = t
        self.start_date = s
        self.end_date = e
        self.historical_prices_USD = self.__fetch_historical_prices()
        self.historical_prices = self.__historical_aggre_return()
        self.avg_daily_return, self.avg_daily_return_sigma = self.__calculate_mu_sigma()
        # Store each parameter as a data member and initialize the variables for the stock class
        
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
        self.historical_prices_USD = self.__fetch_historical_prices()
        self.historical_prices = self.__historical_aggre_return()
        self.avg_daily_return, self.avg_daily_return_sigma = self.__calculate_mu_sigma(self.historical_prices)

    def plot_historical_prices(self):
        plt.figure(figsize = (10,8))
        plt.plot(self.historical_prices)
        plt.xlabel('Date')
        plt.ylabel('Percent Aggregate Return') 
        plt.title('Historical Aggregate Return')
        plt.show()
    
    def __fetch_historical_prices(self):
        """
        Fetches the stock data for historical prices using the Polygon.io API

        Returns:
        historical_prices_USD: np array 
            The closed prices during the chosen time period 
        """
        # load_dotenv()
        # api_key = os.getenv('API_KEY')
        api_key = 'yNTTUyB6GMUYWZFOn6_xTjc18U4jGLj3'
        url = f'https://api.polygon.io/v2/aggs/ticker/{self.ticker}/range/1/day/{self.start_date}/{self.end_date}?adjusted=true&sort=asc&limit=120&apiKey={api_key}'
        
        response = requests.get(url)
        data = response.json()
        if 'results' in data:
            df = pd.DataFrame(data['results'])
            historical_prices_USD = df['c']
            # Convert the list to a numpy array
            if not isinstance(historical_prices_USD, np.ndarray):
                historical_prices_USD = np.array(historical_prices_USD)
            return historical_prices_USD
        else:
            print("No data available for the specified dates and ticker.")
    
    def __calculate_mu_sigma(self):
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
    
    def __historical_aggre_return(self):
        """
        Calculate the aggregated daily average returns from an array of closed stock prices.

        Returns:
        historical_prices: np array
            A list of aggregated daily average returns with 100 as baseline starting point
        """
        # Calculate aggregated daily returns
        # Using the formula: (P_t - P_0) / P_0, where P_t is the price at time t
        daily_returns = (self.historical_prices_USD[1:] - self.historical_prices_USD[0]) / self.historical_prices_USD[0]
        # Adding 0 at the start for baseline for day 1 of observation
        adjusted_daily_returns = np.insert(daily_returns, 0, 0)
        # Making the list into percentage changes
        adjusted_daily_returns = (adjusted_daily_returns + 1) * 100
        
        return np.array(adjusted_daily_returns)