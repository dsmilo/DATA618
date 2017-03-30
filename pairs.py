"""
DATA 618 Project 1: Pairs Trading
Dan Smilowitz
March 29, 2017

This algorithm implements a pairs-trading strategy with large-cap US power producers.
For each potential pair of the identified stocks, the algorithm performs the following:
  - checks history of stocks in pair
  - checks for cointegration between stocks
  - identifies if the stocks have moved from their stationary relationship (over 1 sd)
  - buys or sells based on direction of change
  - cancels existing positions once equilibrium has returned

The algorithm can be manipulated to look back for different number of days and
allocate different amounts of the cash in portfolio used for executing the strategy.

Optimum values were achieved by allocating 50% of available cash for each
cointegrated pair and gathering 15 days of history.

Running a backtest from 1/1/2010 - 3/28/2017 with $1,000,000 in capital
yields returns of 16.6%, a Beta of -0.04, an Alpha of 0.03 and a Sharpe ratio of 0.26.
"""

from statsmodels.tsa.stattools import adfuller
from scipy import stats
import numpy as np
import pandas as pd

# dictionary of stocks to consider:
# US-based power generation public utilities on NYSE with market cap > $20B
# from http://www.nasdaq.com/screening/companies-by-industry.aspx
# note: EXCU excluded due to co-ownership with EXC
# {'ticker': [sid, [list for history]]}

energy = {'ED': sid(2434),
          'DUK': sid(2351),
          'EXC': sid(22114),
          'PCG': sid(5792),
          'PEG': sid(5862),
          'XEL': sid(21964)}

# container for pairs
pairs = [] # (x_ticker, y_ticker, [selling_x, selling_y])

# algorithm parameters to allow for optimization
num_days = 15
cash_share = 0.5


def initialize(context):
    '''
    Initialize stocks of interest and possible pairs (10)
    '''
    global energy
    global pairs
    #context.stocks = stocks
    # get all possible pairings of stocks (order doesn't matter)
    symbols = energy.keys()
    for i in range(len(symbols) - 1):
        for j in range(i + 1, len(symbols)):
            pairs.append((symbols[i], symbols[j], [False, False]))
    # save all pairs to log
    log.info(str(len(pairs)) + " pairs: " + str([(pair[0], pair[1]) for pair in pairs]))
    # run algorithm twice per day: 1 hour after open & one hour before close
    schedule_function(pair_trade_algo,
                      date_rules.every_day(),
                      time_rules.market_open(hours=1, minutes=0))
    schedule_function(pair_trade_algo,
                      date_rules.every_day(),
                      time_rules.market_close(hours=1, minutes=0))


def stationary(x, p=0.05):
    '''
    function to check if x ~ I(0) at significance p using augmented dickey-fuller
    '''
    df = adfuller(x, regression='ctt')
    # check p-value of ADF test statistic for significance
    return df[1] < p


def cointegrated(x, y):
    '''
    function to check if x & y are cointegrated using engle granger
    '''
    if stationary(x) or stationary(y):
        # cointegration is not possible if x or y are stationary
        return False
    else:
        # calculate beta and constant; calculate u
        (beta, c) = stats.linregress(x, y)[0:2]
        u = y - x * beta - c
        # x & y are cointegrated if u is stationary
        return stationary(u)


def pair_trade_algo(context, data):
    '''
    function to execute pairs trading strategy for each pair
    bases cointegration and deviation on num_days trading days
    allocates cash_share of total cash for trading strategy
    '''
    global pairs
    global energy
    global num_days
    global cash_share

    # local function for log returns
    def log_prices(x, days=1):
        '''
        function to return log-normalized prices of security x for given days
        '''
        # get Quantopian data
        d = data.history(x, 'price', days, '1d')
        # log prices
        return np.log10(d.values)

    # execute for each possible pair
    for pair in pairs:
        # stock sid, current price, history
        x_sid = energy[pair[0]]
        y_sid = energy[pair[1]]
        x = np.log10(data.current(x_sid, 'price'))
        y = np.log10(data.current(y_sid, 'price'))
        x_h = log_prices(x_sid, num_days)
        y_h = log_prices(y_sid, num_days)
        # stock difference history, mean, std, current difference
        diff_h = x_h - y_h
        sd = np.std(diff_h)
        mean = np.average(diff_h)
        diff = x - y
        # check current difference relative to historical
        z = (diff - mean) / sd

        # if >1 sd from mean and cointegrated, peform trade
        if abs(z) > 1 and cointegrated(x_h, y_h):
            # number of shares to buy and sell
            # allocate cash_share evenly between pair
            cash_to_trade = context.portfolio.cash * cash_share
            x_shares = (cash_to_trade / 2) / data.current(x_sid, 'price')
            y_shares = (cash_to_trade / 2) / data.current(y_sid, 'price')
            # if high x causes deviation, sell x & buy y (if not already)
            if z > 1 and pair[2][0] == False:
                log.info("Selling %d shares of %s and buying %d shares of %s"
                         %(x_shares, pair[0], y_shares, pair[1]))
                # execute trades
                order(x_sid, -x_shares)
                order(y_sid, y_shares)
                # note current position in pair
                pair[2][0] = True
                pair[2][1] = False
            # if high y causes deviation, sell y & buy x (if not already)
            elif z < -1 and pair[2][1] == False:
                log.info("Selling %d shares of %s and buying %d shares of %s"
                         %(y_shares, pair[1], y_shares, pair[0]))
                # execute trades
                order(x_sid, x_shares)
                order(y_sid, -y_shares)
                # note current position in pair
                pair[2][0] = False
                pair[2][1] = True
        # if mean reversion, close open positions
        elif abs(z) < 1 and pair[2][0] or pair[2][1]:
            log.info("Closing open positions of %s and %s" %(pair[0], pair[1]))
            # close current position (either by buying or selling)
            order_target(x_sid, 0)
            order_target(y_sid, 0)
            # reflect reset positions
            pair[2][0] = False
            pair[2][1] = False
