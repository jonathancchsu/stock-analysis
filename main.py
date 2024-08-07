from stock import *
from monte_carlo import *
from FCN import *
from utilities import *

# download data, store in stock_data folder
# data_start_date = '2016-01-02'
# data_end_date = '2024-07-30'

# Utilities.download_data(data_start_date, data_end_date, 'NVDA')
# Utilities.download_data(data_start_date, data_end_date, 'SMCI')
# Utilities.download_data(data_start_date, data_end_date, 'TSM')

start_date = '2021-11-05'
end_date = '2022-11-04'
single_date = '2022-11-04'

# Stock params: ticker, days to calculate sigma mu
NVDA = Stock('NVDA', 250)
SMCI = Stock('SMCI', 250)
TSM = Stock('TSM', 250)

stocks = [NVDA, SMCI, TSM]

# fcn params: list of Stocks, ko (default=120), ki (default=60), strike (default=80),
#         guaranteed_days (default=27), tenor (default=252), risk-free rate(default-0.048)
fcn = FCN(stocks, 110, 70, 90, 27, 252, 0.048)

# backtest_fcn params: start_date('YYYY-MM-DD'), end_date('YYYY-MM-DD')
# fcn.backtest_fcn(start_date, end_date)

# backtest_fcn_single_date params: date('YYYY-MM-DD')
# fcn.backtest_fcn_single_date(single_date)

# simulate_fcn params: num_trials, start_days_from_today, number of days generated, 
#                      correlated(default=True), number of graphs generated(default=5)
# fcn.simulate_fcn(1000, 0, 500, True, 5)
fcn.simulate_fcn(1000, 0, 500, False, 0)


######################Functions for Testing######################
# generate_correlated_random_walks params: list of stock simulations, length of each 
# path (in days), risk free rate
# random_walks = fcn.generate_correlated_random_walks(500)
# fcn.graph_random_walks(random_walks, 500)

# MCS params: Stock, maturity (default=252), periods (default=2) 
# NVDAsim = MCStockSimulator(NVDA, 252, 2)
# SMCIsim = MCStockSimulator(SMCI, 252, 2)
# TSMsim = MCStockSimulator(TSM, 252, 2)

# generate_simulated_stock_values params: num_trials (default=10)
# NVDAsim_vals = NVDAsim.generate_simulated_stock_values(10)

# plot_simulated_stock_values params: simulated stock values
# NVDAsim.plot_simulated_stock_values(NVDAsim_vals)