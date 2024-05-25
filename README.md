# Stock-Analysis (for Fixed Coupon Notes)
---
**Using Monte-Carlo method to simulate the paths of given stocks and time frames.**
### Class Designs
 - Stock class:
   - Generate historical aggregated return prices, average daily return, and standard deviation of the average daily return given a ticker and a timeframe.
 - MCStockSimulator class:
   - Capabilities to generate simulated stock returns and values given a stock and capabilities to plot the simulations.
### Functions
 - plot_stock_values:
   - Plotting the percent changes of stock(s) (baseline 100 using the price on the first day).
