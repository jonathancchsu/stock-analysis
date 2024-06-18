from stock import *
from monte_carlo import *
from FCN import *
from utilities import *

# download data, store in stock_data folder
# data_start_date = '2019-01-02'
# data_end_date = '2024-06-18'

# Utilities.download_data(data_start_date, data_end_date, 'NVDA')

start_date = '2021-01-04'
end_date = '2023-06-05'

# Stock params: ticker
NVDA = Stock('NVDA')
SMCI = Stock('SMCI')
TSM = Stock('TSM')

# MCS params: Stock, maturity (default=252), periods (default=2) 
NVDAsim = MCStockSimulator(NVDA, 252, 2)
SMCIsim = MCStockSimulator(SMCI, 252, 2)
TSMsim = MCStockSimulator(TSM, 252, 2)

# generate_simulated_stock_values params: num_trials (default=10)
# NVDAsim_vals = NVDAsim.generate_simulated_stock_values(10)

# plot_simulated_stock_values params: simulated stock values
# NVDAsim.plot_simulated_stock_values(NVDAsim_vals)

stocks = [NVDA, SMCI, TSM]
stock_sims = [NVDAsim, SMCIsim, TSMsim]
# fcn params: list of Stocks, ko (default=120), ki (default=60), strike (default=80),
#         guaranteed_days (default=27), tenor (default=252)  
fcn = FCN(stocks, 120, 60, 80, 27, 252)

ko_dates = fcn.backtest_KO(start_date, end_date)
print(ko_dates)
fcn.graph_backtest_KO(start_date, end_date, ko_dates)

# simulate_KO params: stock value simulations, num_trials (default=1000), 
#                     start_days_from_today (default=0)
print(fcn.simulate_KO(stock_sims, 1000, 0))