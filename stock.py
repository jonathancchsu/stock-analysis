import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Stock:
    """
    The Stock class is designed to contain the information of a Stock given the ticker
    and the given time frame using Polygon.api
    """

    def __init__(self, t):
        """
        Parameters
        ----------
        t : string
            The ticker of the stock, e.g. 'NVDA'

        Initialized Variables (eager loading)
        ----------
        historical_prices: pandas df
            Contains the dates, historical closing prices, and historical aggregated returns of the stock
        """

        self.ticker = t
        self.historical_prices = pd.DataFrame()
        self.fetch_historical_prices()
        self.historical_aggre_return()
        
    def __repr__(self):
        """Returns a formatted string containing the data members of the class"""

        info = (f'Stock (t=${self.ticker},' 
                f'historical_prices={self.historical_prices['Close']})')
        return info
    

    def calculate_aggre_returns(self, start_date, end_date):
        """
        Calculate the aggregated returns in a given time frame.
        """
        # Calculate aggregated daily returns
        # Using the formula: (P_t - P_0) / P_0, where P_t is the price at time t
        start_date = pd.to_datetime(start_date).date()
        end_date = pd.to_datetime(end_date).date()
        df = self.historical_prices[['Date' , 'Close']]
        filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
        filtered_df = filtered_df.reset_index(drop=True)
        filtered_returns = (filtered_df['Close'][1:] - filtered_df['Close'][0]) / filtered_df['Close'][0]
        # Adding 0 at the start for baseline for day 1 of observation
    
        filtered_returns = np.insert(filtered_returns, 0, 0)
        # Making the list into percentage changes
        filtered_returns = (filtered_returns + 1) * 100
        filtered_returns = np.array(filtered_returns)

        return filtered_returns

    def plot_historical_prices(self):
        print(self.historical_prices)
        plt.figure(figsize = (10,8))
        plt.plot(self.historical_prices['aggregated_historical_returns'])
        plt.xlabel('Date')
        plt.ylabel('Percent Aggregate Return') 
        plt.title(f'Historical Aggregate Return of {self.ticker}')
        plt.show()
    
    def fetch_historical_prices(self):
        """
        Fetches the stock data for historical prices using the Polygon.io API

        Returns:
        historical_prices: np array 
            The closed prices during the chosen time period 
        """

        data = pd.read_csv(f'stock_data/{self.ticker}.csv')
        # print(data)
        df = data[['Date' , 'Close']].copy()
        # print(df)
        df['Close'] = np.array(df['Close'])
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
        self.historical_prices = df
    
    def historical_aggre_return(self):
        """
        Get the aggregated daily average returns from an array of closed stock prices.

        Returns:
        historical_prices: np array
            A list of aggregated daily average returns with 100 as baseline starting point
        """
        start_date = self.historical_prices['Date'].iloc[0]
        end_date = self.historical_prices['Date'].iloc[-1]
        adjusted_daily_returns = self.calculate_aggre_returns(start_date, end_date)

        df_aggregated_daily_returns = pd.DataFrame({'aggregated_historical_returns': adjusted_daily_returns})
        df = pd.concat([self.historical_prices, df_aggregated_daily_returns], axis=1)


        self.historical_prices = df