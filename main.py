# -*- coding: utf-8 -*-
from plot_stock_values import *
from stock import *
from monte_carlo import *

start_date = '2022-06-30'
end_date = '2023-06-30'

NVDA = Stock('NVDA', start_date, end_date)
SMCI = Stock('SMCI', start_date, end_date)
TSM = Stock('TSM', start_date, end_date)

NVDAsim = MCStockSimulator(NVDA)
SMCIsim = MCStockSimulator(SMCI)
TSMsim = MCStockSimulator(TSM)

NVDAsim.plot_simulated_stock_values()
SMCIsim.plot_simulated_stock_values()
TSMsim.plot_simulated_stock_values()

stocks = [NVDA, SMCI, TSM]

plot_stock_values(stocks)