import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Timestamp

class Stock:
    """
    The Stock class is designed to contain the information of a Stock given the ticker
    the data of the stock downloaded from yfinance API into csv files and calculate the 
    historical aggregate return, rate of return, and sigma of the stock given the data and
    specified training days
    """

    def __init__(self, t, train=758):
        """
        Parameters:
        t : string
            The ticker of the stock, e.g. 'NVDA'

        Initialized Variables (eager loading)
        historical_prices: pandas df
            Contains the dates, historical closing prices, and historical aggregated 
            returns of the stock
        rate_of_return, sigma: float
            The average daily return and the standard deviation of the average daily return
            of the stocks based on the given amount of dates.
        """

        self.ticker = t
        self.historical_prices = pd.DataFrame()
        self.train = train
        self.fetch_historical_prices()
        self.historical_aggregate_return()
        self.rate_of_return, self.sigma = self.calculate_mu_sigma()
        
    def __repr__(self):
        """Returns a formatted string containing the ticker and closing prices of the stock"""

        info = (f'Stock (t=${self.ticker},' 
                f'historical_prices={self.historical_prices['Close']})')
        return info
    

    def calculate_aggregate_returns(self, start_date, end_date):
        """
        Calculate the aggregated returns in a given time frame.

        Parameters:
        start_date: string 
            The start date in YYYY-MM-DD format.
        end_date: string 
            The end date in YYYY-MM-DD format.

        Returns:
            Percent of aggregated daily returns in a list given the timeframe.
        """
        # Convert dates to pandas.Timestamp directly
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Make sure the Date column is in Timestamp format
        self.historical_prices['Date'] = pd.to_datetime(self.historical_prices['Date'])

        # Filter DataFrame safely using .loc and explicit copy
        filtered_df = self.historical_prices.loc[(self.historical_prices['Date'] >= start_date) & (self.historical_prices['Date'] <= end_date)].copy()
        filtered_df.reset_index(drop=True, inplace=True)

        # Calculate the returns
        if len(filtered_df) > 1:
            filtered_returns = (filtered_df['Close'][1:] - filtered_df['Close'].iloc[0]) / filtered_df['Close'].iloc[0]
        else:
            filtered_returns = np.array([])  # handle case where filtered_df is too short

        # Prepare returns for output
        filtered_returns = np.insert(filtered_returns, 0, 0)  # add 0 at the start for baseline
        filtered_returns = (filtered_returns + 1) * 100  # convert to percentage

        return filtered_returns

    def plot_historical_prices(self):
        """Plot the graph of the stock according to the historical prices."""
        plt.figure(figsize = (10,8))
        plt.plot(self.historical_prices['aggregated_historical_returns'])
        plt.xlabel('Date')
        plt.ylabel('Percent Aggregate Return') 
        plt.title(f'Historical Aggregate Return of {self.ticker}')
        plt.show()
    
    def fetch_historical_prices(self):
        """
        Fetches the stock data for dates and historical prices from the csv file 
        in stock_data folder

        Returns:
        historical_prices: np array 
            The closed prices during the chosen time period 
        """

        data = pd.read_csv(f'stock_data/{self.ticker}.csv')
        df = data[['Date' , 'Close']].copy()
        df['Close'] = np.array(df['Close'])
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
        self.historical_prices = df
    
    def historical_aggregate_return(self):
        """
        Get the historical aggregated daily returns of the stock, attach to the 
        historical_prices df
        """
        start_date = self.historical_prices['Date'].iloc[0]
        end_date = self.historical_prices['Date'].iloc[-1]
        adjusted_daily_returns = self.calculate_aggregate_returns(start_date, end_date)

        df_aggregated_daily_returns = pd.DataFrame({'aggregated_historical_returns': adjusted_daily_returns})
        df = pd.concat([self.historical_prices, df_aggregated_daily_returns], axis=1)


        self.historical_prices = df

    def calculate_mu_sigma(self):
        """
        Calculate the daily average return and the standard deviation from an array of closed stock prices.

        Returns:
        average_daily_return: float
            The daily average return as a percentage.
        avg_daily_return_sigma: float 
            The standard deviation of the daily returns.
        """

        past_close = self.historical_prices['Close'].iloc[-self.train:]
        past_close = past_close.reset_index(drop=True).to_numpy()
        # Calculate daily returns
        # Using the formula: (P_t - P_t-1) / P_t-1, where P_t is the price at time t
        daily_returns = (past_close[1:] - past_close[:-1]) / past_close[:-1]
        # Calculate the average of these daily returns
        average_return = np.mean(daily_returns)

        return average_return, daily_returns.std()