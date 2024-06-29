# Stock-Analysis (for Fixed Coupon Notes)
---
***Using past data to backtest and Monte-Carlo method to simulate the different scenarios of FCN.***
### Class Designs
 - **Stock class**:
   - Fetches the historical prices of a stock given a ticker and calculate the mu and sigma based on the given time.
   - Functions:
     - **Stock(ticker, train=758)**
     - calculate_aggregate_returns(start_date, end_date)
     - plot_historical_prices()
     - fetch_historical_prices()
     - historical_aggregate_return()
     - calculate_mu_sigma()

 - **FCN class**:
   - Backtest with past data and gain information based on simulated data for analyzing the different scenarios of FCN.
   - Functions:
      - **FCN(stocks, ko=120, ki=60, strike=80, guaranteed_days=27, tenor=252, risk_free_rate=0.048)**
      - **backtest_fcn(start_date, end_date)**
      - **backtest_fcn_single_date(date)**
      - **simulate_fcn(num_trials=1000, start_days_from_today=0, time_steps=500)**
      - plot_stocks()
      - check_dates(start_date, end_date)
      - graph_backtest_fcn(start_date, end_date)
      - calculate_correlation(days)
      - generate_correlated_random_walks(time_steps)
      - graph_random_walks(stock_prices, time_steps)
      - print_dates(dates_dict)

 - **Utilities class**:
   - Utility functions
   - Functions:
     - **download_data(start_date, end_date, ticker)**

 - **MCStockSimulator class (deprecated)**:
   - Capabilities to generate simulated single stock returns and values given a stock and plot the simulations.
   - Functions:
     - **MCStockSimulator(Stock, maturity_time=252, periods_per_year=2, train=758)**
     - **generate_simulated_stock_values(num_trials=10)**
     - **plot_simulated_stock_values(simulated_stock_values)**
     - calculate_mu_sigma()
     - generate_simulated_stock_returns()