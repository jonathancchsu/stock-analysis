# -*- coding: utf-8 -*-
from stock import *
from monte_carlo import *
from FCN import *

start_date = '2021-01-04'
end_date = '2023-06-05'

NVDA = Stock('NVDA')
SMCI = Stock('SMCI')
TSM = Stock('TSM')

# NVDAsim = MCStockSimulator(NVDA)
# SMCIsim = MCStockSimulator(SMCI)
# TSMsim = MCStockSimulator(TSM)

# NVDAsim.plot_simulated_stock_values(50)
# SMCIsim.plot_simulated_stock_values(50)
# TSMsim.plot_simulated_stock_values(50)

stocks = [NVDA, SMCI, TSM]

# for stock in stocks:
#   stock.plot_historical_prices()
fcn = FCN(stocks, 120, 60, 80, 27, 252)

# fcn.simulate_KO
ko_dates = fcn.backtest_KO(start_date, end_date)
print(ko_dates)
fcn.graph_backtest_KO(start_date, end_date, ko_dates)

# fcn.plot_stocks()