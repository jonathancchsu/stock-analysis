# -*- coding: utf-8 -*-
from stock import *
from monte_carlo import *
from FCN import *

start_date = '2022-06-14'
end_date = '2023-06-14'

NVDA = Stock('NVDA', start_date, end_date)
SMCI = Stock('SMCI', start_date, end_date)
TSM = Stock('TSM', start_date, end_date)

# NVDAsim = MCStockSimulator(NVDA)
# SMCIsim = MCStockSimulator(SMCI)
# TSMsim = MCStockSimulator(TSM)

# NVDAsim.plot_simulated_stock_values(50)
# SMCIsim.plot_simulated_stock_values(50)
# TSMsim.plot_simulated_stock_values(50)

stocks = [NVDA, SMCI, TSM]

fcn = FCN(stocks, 120, 110, 100, 27, 252)

ko_dates = fcn.backtest_KO(start_date, end_date)
print(ko_dates)
fcn.graph_backtest_KO(start_date, end_date, ko_dates)

# fcn.plot_stocks()