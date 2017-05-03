"""
DATA 618 Project 2: Financial Machine Learning
Dan Smilowitz
April 26, 2017

This algorithm implements a trading strategy based on machine learning algorithms.
Two large-cap US power producers are considered in this algorithm, with each
security's transactions based on a different ML algorithm:
  - DUK (Duke Energy): Random Forest Classifier
  - PCG (Pacific Gas & Electric): Gaussian Naive Bayes

Throughout the day (1 minute after open, 1 minute before close, and at every half-hour),
the algorithm performs the following steps for each stock/algorithm:
  - gets previous num_bars minutes of data for variables my_chars
  - checks if each variable increased from its value the previous minute
  - trains the models on the incrase/decrease data
    - price is dependent variable
  - gets the current value of independent variables
  - using trained model, predicts if price will increase
  - if increase is predicted, buys stock
    - uses half of the cash_share allocated for the strategy for each stock
  - if no increase predicted, sells all shares of stock

The algorithm can be manipulated in the following ways:
  - the characterstics considered my_chars
  - the number of minutes used for training num_bars
  - the share of cash used for the strategy (half to each stock) cash_share

Optimum values were achieved by considering volume, high, and low as features,
allocating 25% of available cash for the strategy, and training on 10 minutes of history.

Running a backtest from 1/1/2010 - 4/25/2017 with $1,000,000 in capital
yields returns of -26.2%, an Alpha of -0.05 and a Sharpe ratio of -1.74.

This is significantly worse than the performance of a portfolio comprised of 50%
of each stock, which yielded returns of 88.6%, Alpha of 0.05, and Sharpe of 0.74.
This suggests there is serious room for improvement, likely through the incorporation
of additional features into the models used.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd

# dictionary of stocks to consider:
# US-based power generation public utilities on NYSE with market cap > $20B
# from http://www.nasdaq.com/screening/companies-by-industry.aspx
# note: EXCU excluded due to co-ownership with EXC
# {'ticker': sid}

energy = {'DUK': sid(2351),
          'PCG': sid(5792),
          'EXC': sid(22114),
          'ED': sid(2434),
          'XEL': sid(21964),
          'PEG': sid(5862)}

# algorithm parameters to allow for optimization
my_chars = ['price', 'volume', 'high', 'low']    # characterstics to feed into model
num_bars = 10       # number of bars to consider
cash_share = 0.25    # share of available cash to use for strategies

def initialize(context):
    '''
    Initialize stocks and ML algorithms; schedule function
    '''
    # stocks for use -- two largest energy stocks above (Duke Energy & Pacific Gas and Electric)
    global energy
    context.stock1 = energy['DUK']
    context.stock2 = energy['PCG']
    # initialization of ML algorithms
    context.forest = RandomForestClassifier()
    context.y1 = 0
    context.bayes = GaussianNB()
    context.y2 = 0
    # run trading algorithm 1 minute after open
    schedule_function(ml_trade_algo,
                      date_rules.every_day(),
                      time_rules.market_open(minutes=1))
    # run trading algorithm at every half hour
    for t in range(30, int(6.5*60), 30):
        schedule_function(ml_trade_algo,
                          date_rules.every_day(),
                          time_rules.market_open(minutes=t))
    # run trading algorithm 1 minute before close
    schedule_function(ml_trade_algo,
                      date_rules.every_day(),
                      time_rules.market_close(minutes=1))

def ml_trade_algo(context, data):
    global my_chars
    global num_bars
    # get historical price & volume information
    df_hist1 = data.history(context.stock1, my_chars, num_bars, '1m')
    df_hist2 = data.history(context.stock2, my_chars, num_bars, '1m')
    # convert to df seeing if increased
    df_diff1 = pd.DataFrame(data=np.diff(df_hist1, axis=0) > 0, columns=my_chars)
    df_diff2 = pd.DataFrame(data=np.diff(df_hist2, axis=0) > 0, columns=my_chars)
    # convert to independent & dependent variables -- introduce lag
    X1 = df_diff1[:-1].drop('price', axis=1)
    Y1 = df_diff1.ix[1:, 'price']
    X2 = df_diff2[:-1].drop('price', axis=1)
    Y2 = df_diff2.ix[1:, 'price']
    # train the models
    context.forest.fit(X1, Y1)
    context.bayes.fit(X2, Y2)
    # get values for prediction
    x1 = data.current(context.stock1, [c for c in my_chars if c != 'price'])
    x2 = data.current(context.stock2, [c for c in my_chars if c != 'price'])
    # get predictions
    try:
        y1 = context.forest.predict(x1).item()
    except Exception:
        y1 = 0
    try:
        y2 = context.bayes.predict(x2).item()
    except Exception:
        y2 = 0
    # record predictions
    record(Stock1_Prediction = y1, Stock2_Prediction = y2)
    # execute trades
    if y1:
        order_target_percent(context.stock1, context.cash_share / 2)
        log.info('Predicted increase in stock 1; buying')
    else:
        order_target_percent(context.stock1, 0)
        log.info('Stock 1 not increasing; selling')
    if y2:
        order_target_percent(context.stock2, context.cash_share / 2)
        log.info('Predicted increase in stock 2; buying')
    else:
        order_target_percent(context.stock2, 0)
        log.info('Stock 2 not increasing; selling')
