# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import exp
from stock import *


class MCStockSimulator:
    """
    The MCStockSimulator class is designed to represent a Monte Carlo simulation
    of a stock over the period of an option's life time. The parameters required
    are the current stock price, the time to maturity of the option, the average
    annual rate of return of the stock, the annual standard deviation of the returns,
    and the number of periods per year.
    """
    
    def __init__(self, stock, t=252, p=2, train=758):
        """
        Parameters: 
        stock : Stock object
            contains the ticker and historical prices data
        t : float 
            The time to maturity of the option in days, alternatively the time
            horizon we wish to model
        p : float
            The number of periods per year
        train: float 
            The amount of trading days we wish to trade on 
        Other Variables:
        rate_of_return: float 
            The average daily return of the stock
        sigma: float 
            The standard deviation of the average daily return of the stock
        ticker: string
            Ticker of the stock
        stock_price: float
            The default baseline of the stock 
        """

        self.stock = stock
        self.maturity_time = t
        self.train = train
        self.rate_of_return, self.sigma = self.calculate_mu_sigma()
        self.periods_per_year = p
        self.ticker = stock.ticker
        self.stock_price = 100
    
    def __repr__(self):
        """Returns a formatted string containing the data members of the class"""
        
        info = (f'StockSimulator (s=${self.stock_price:.2f}, t={self.maturity_time:.2f} '
                f'(days), rate_of_return={self.rate_of_return:.2f}, sigma={self.sigma:.2f} '
                f'p={self.periods_per_year:.0f}, r={self.repetitions:.0f})')
        return info
    
    def calculate_mu_sigma(self):
        """
        Calculate the daily average return and the standard deviation from an array of closed stock prices.

        Returns:
        average_daily_return: float
            The daily average return as a percentage.
        avg_daily_return_sigma: float 
            The standard deviation of the daily returns.
        """

        past_close = self.stock.historical_prices['Close'].iloc[-self.train:]
        past_close = past_close.reset_index(drop=True).to_numpy()
        # Calculate daily returns
        # Using the formula: (P_t - P_t-1) / P_t-1, where P_t is the price at time t
        daily_returns = (past_close[1:] - past_close[:-1]) / past_close[:-1]
        # Calculate the average of these daily returns
        average_return = np.mean(daily_returns)

        return average_return, daily_returns.std()
    
    def generate_simulated_stock_returns(self):
        """
        Generates and returns an numpy.array of simulated stock returns for the
        time period and number of periods per year
        Returns: 
            simulation: list[float]
                a list of simulated stock values 
        """
        
        length = int(self.maturity_time * self.periods_per_year)
        # the length of the array is the time in years times the number of periods
        # per year. Change the data type to int for the random.normal() function.
        
        dt = 1 / self.periods_per_year 
        # calculate the length of each discrete time period 
        
        simulation = ((self.rate_of_return - (self.sigma**2 / 2)) * dt + 
                       np.random.normal(size = length) * self.sigma * dt**0.5)
        # each return is simulated by taking the mean rate of return multiplied
        # by the variance divided by 2, multiplied by dt plus a random number
        # times the standard deviation fo returns times the square root of dt
                        
        return simulation
    
    def generate_simulated_stock_values(self, num_trials=10):
        """
        Generates and returns an numpy.array of simulated stock values for the
        time period and number of periods per year
        Parameters:
            num_trials: int 
                The number of simulations ran 
        Returns:
            list_of_prices: list[np.array]
                A list of np.arrays that contains the simulated stock prices
        """
        
        list_of_prices = []  # Use a list to store arrays of prices for each trial

        for _ in range(num_trials):
            returns = self.generate_simulated_stock_returns()
            prices = np.array([self.stock_price])  # Initialize the array with the initial stock price

            for I in range(len(returns)):
                next_price = prices[-1] * exp(returns[I])  # Calculate the next price
                prices = np.append(prices, next_price)  # Append the next price to the prices array
                # for each item in the returns array, multiply it by the price at the
                # same index and append it to the prices array. This generates the
                # array of prices. 
            list_of_prices.append(prices)  # Append the complete prices array for this trial to the list

        return list_of_prices
    
    def plot_simulated_stock_values(self, simulated_stock_values):
            """
            Plots the simulated stock values on a graph.
            """
            x_axis = np.linspace(0, self.maturity_time, 
                                 num = self.periods_per_year * self.maturity_time + 1)
            # Create the x axis. The length should be the periods per year times
            # the time to maturity plus one for the current stock price
            
            # fig = plt.figure()
            plt.figure(figsize = (10,8))

            # Create the graph and label the axes
            
            trials_count = len(simulated_stock_values)
            for i in range(trials_count):
                prices = simulated_stock_values[i]
                plt.plot(x_axis, prices)
                # For each trial, generate a simulated stock values array and plot it
                # with the x axis
            ticker = self.ticker
            plt.title(f'{trials_count} simulated trials for {ticker}')
            plt.xlabel(f'Days')
            plt.ylabel(f'$ Value')