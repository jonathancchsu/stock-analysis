# Stock-Analysis (for Fixed Coupon Notes)
---
**Using past data to backtest and Monte-Carlo method to simulate the different scenarios of FCN.**
### Class Designs
 - Stock class:
   - Fetches the historical aggregated return prices given a ticker 
   - Functions:
     - Stock('ticker')
     - calculate_aggre_returns(start_date, end_date)
     - plot_historical_prices()
     - fetch_historical_prices()
     - historical_aggre_return()
 - MCStockSimulator class:
   - Capabilities to generate simulated stock returns and values given a stock and plot the simulations.
   - Functions:
     - MCStockSimulator(Stock, maturity_time=252, periods_per_year=2, train=758)
      - generate_simulated_stock_returns()
      - generate_simulated_stock_values(num_trials=10)
      - plot_simulated_stock_values(simulated_stock_values)
 - FCN class:
   - Backtest with past data and gain information based on simulated data for analyzing the different scenarios of FNC.
   - Functions:
    - FCN(stocks, ko=120, ki=60, strike=80, guaranteed_days=27, tenor=252)
    - plot_stocks()
    - check_dates(start_date, end_date)
    - backtest_KO(start_date, end_date)
    - graph_backtest_KO(start_date, end_date, ko_dates)
    - simulate_KO(list_of_simulations, num_trials=1000, start_days_from_today=0)
    - backtest_KI(start_date, end_date) (in progress)
    - graph_backtest_KI(start_date, end_date, ki_dates) (in progress)
    - simulate_KI(list_of_simulations, num_trials=1000, start_days_from_today=0) (in progress)
    - backtest_exercise(self, start_date, end_date) (in progress)
    - graph_backtest_exercise(start_date, end_date, exercise_dates) (in progress)
    - simulate_exercise(list_of_simulations, num_trials=1000, start_days_from_today=0) (in progress)