import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import yfinance as yf

def visualize_signals(signals):
  # Initialize the plot figure
  fig = plt.figure( figsize=(8, 6))

  # Add a subplot and label for y-axis
  ax1 = fig.add_subplot(111,  ylabel='Price in $')

  # Plot the closing price
  # aapl['Close'].plot(ax=ax1, color='r', lw=2.)

  # Plot the short and long moving averages
  signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

  # Plot the buy signals
  ax1.plot(signals.loc[signals.positions == 1.0].index, 
          signals.short_mavg[signals.positions == 1.0],
          '^', markersize=10, color='m')
          
  # Plot the sell signals
  ax1.plot(signals.loc[signals.positions == -1.0].index, 
          signals.short_mavg[signals.positions == -1.0],
          'v', markersize=10, color='k')
          
  # Show the plot
  plt.show()

def load_ticker(ticker):
  yf.pdr_override()
  stock = pdr.get_data_yahoo(ticker, 
                            start=datetime.datetime(2006, 10, 1), 
                            end=datetime.datetime(2021, 12, 5))
  return stock

def ma_strategy(stock, short_window=40, long_window=100):
  # Initialize the short and long windows# Initia 

  # Initialize the `signals` DataFrame with the `signal` column
  signals = pd.DataFrame(index=stock.index)
  signals['signal'] = 0.0

  # Create short simple moving average over the short window
  signals['short_mavg'] = stock['Close'].rolling(window=short_window, min_periods=1, center=False).mean()

  # Create long simple moving average over the long window
  signals['long_mavg'] = stock['Close'].rolling(window=long_window, min_periods=1, center=False).mean()

  # Create signals
  signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:] 
                                              > signals['long_mavg'][short_window:], 1.0, 0.0)   

  # Generate trading orders
  signals['positions'] = signals['signal'].diff()
  return signals

def portfolio_evaluation(signals, stock):
  np.where(signals.signal.diff() == 1)
  np.where(signals.signal.diff() == -1)
  buy_mask = np.where(signals.signal.diff() == 1)[0]
  sell_mask = np.where(signals.signal.diff() == -1)[0]
  np.array(stock['Adj Close'][buy_mask])
  np.array(stock['Adj Close'][sell_mask])
  buy_close = np.array(stock['Adj Close'][buy_mask])
  sell_close = np.array(stock['Adj Close'][sell_mask])
  if (sell_close.shape[0] < buy_close.shape[0]):
    sell_close = np.append(sell_close, stock['Adj Close'].loc['2021-12-03'])
  return (sell_close / buy_close).prod()

def tune_strategy_param(stock, short_window=40, long_window=100):
  signals = ma_strategy(stock, short_window, long_window)
  return portfolio_evaluation(signals, stock)

def init_param_list(short_window_max=100, long_window_max=100):
  param_list = []
  for short_window in np.arange(1, short_window_max):
    for long_window in np.arange(1, long_window_max):
      param_list.append((short_window, long_window))
  return param_list

def evaluate_strategy_performance(stock, param_list):
  return list(map(lambda param : _map_to_strategy_performance(param, stock), param_list))

def visualize_strategy_perf(strategy_perf_mat):
  x = np.array(strategy_perf_mat)[:,0]
  y = np.array(strategy_perf_mat)[:,1]
  z = np.array(strategy_perf_mat)[:,2]
  df = pd.DataFrame.from_dict(np.array([x,y,z]).T)
  df.columns = ['X_value','Y_value','Z_value']
  df['Z_value'] = pd.to_numeric(df['Z_value'])
  pivotted= df.pivot('Y_value','X_value','Z_value')
  ax = sns.heatmap(pivotted,cmap='RdBu')
  ax.invert_yaxis()
  
def draw_stock(stock):
  # Plot the closing prices for `aapl`
  stock['Adj Close'].plot(grid=True)

  # Show the plot
  plt.show()

def draw_stocks(*stocks):
  # Plot the closing prices
  for stock in stocks:
    stock['Adj Close'].plot(grid=True)

  # Show the plot
  plt.show()

def _map_to_strategy_performance(param, stock):
   return (param[0], param[1], tune_strategy_param(stock, param[0], param[1]))