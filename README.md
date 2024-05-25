# Stock-Analysis (for Fixed Coupon Notes)
---
**Using Monte-Carlo method to simulate the paths of given stocks and time frames.**
### Class Designs
 - Stock class:
   - Generate historical aggregated return prices, average daily return, and standard deviation of the average daily return given a ticker and a timeframe.
   - Usage:
     - Stock('ticker', start_date, end_date)
     - get_dates(), 
     - set_dates(s, e) (updates the calculated values), 
     - plot_historical_prices()
 - MCStockSimulator class:
   - Capabilities to generate simulated stock returns and values given a stock and plot the simulations.
   - Usage:
     - MCStockSimulator(Stock, maturity_time (optional, defaults 252), periods_per_year (optional, defaults 252))
      - generate_simulated_stock_returns(),
      - generate_simulated_stock_values(),
      - plot_simulated_stock_values(num_trials (optional, default 1))

### Functions
 - plot_stock_values:
   - Plotting the percent changes of stock(s) (baseline 100 using the price on the first day).
  