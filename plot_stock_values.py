import matplotlib.pyplot as plt

def plot_stock_values(stocks):
    plt.figure(figsize=(10, 8))
    # Ensure labels are present for the legend
    for stock in stocks:
        # You can use stock.ticker or any other unique property of the stock
        plt.plot(stock.historical_prices, label=stock.ticker)
    
    plt.xlabel('Date')
    plt.ylabel('Percent Aggregate Return') 
    plt.title('Historical Aggregate Return')
    plt.legend()  # This will show the legend using the labels defined in plt.plot()
    plt.show()