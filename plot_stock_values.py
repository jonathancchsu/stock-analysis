import matplotlib.pyplot as plt

def plot_stock_values(stocks):
  aggre_prices = []

  for stock in stocks:
    aggre_prices.append(stock.historical_prices)

  plt.figure(figsize = (10,8))
  for i in range(len(aggre_prices)):
    plt.plot(aggre_prices[i])
  plt.xlabel('Date')
  plt.ylabel('Percent Aggregate Return') 
  plt.title('Historical Aggregate Return')
  plt.show()