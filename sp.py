"""
DATA 618 Project 3: Financial Signal Processing
Dan Smilowitz
May 21, 2017

This algorithm implements a trady strategy based on the Kalman filter signal processing
technique.  The strategy builds on a pairs-trading concept using two large-cap US
power producers: EXC (Excelon Energy) and ED (Consolidated Edison).

Throughout the day, the algorithm performs the following steps:
  - gets previous N minutes of price data for each stock
  - uses the data to update the Kalman filter
  - compares the Kalman estimate to the actual price
    - if the difference in the estimate exceeds tolerance bands, opens positions:
      - if under lower band, opens long position on ED hedged with EXC
      - if over upper band, opens short position on ED hedged with EXC
    - closes any open positions when equilibrium returns (difference within bands)

The algorithm can be manipulated in the following ways:
  - the number of minutes considered in the filter N
  - the share of cash used for the strategy (half to each stock) cash_share
  - the frequency at which the algorithm is called

Optimum values were achieved running the algorithm every 15 minutes using N = 30 and cash_share = 1.00.

Running a backtest from 1/1/2010 - 5/19/2017 with $1,000,000 in capital yields
returns of 47.6%, a Beta of 0.00, an Alpha of 0.06 and a Sharpe ratio of 0.92.
"""

import numpy as np

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
N = 50
cash_share = 0.25
algo_freq = 30

# initial Kalman filter parameters
mu = 0.                 # initial state mean
K = np.zeros(N)         # Kalman gain
A = np.zeros(N)         # State-innovation covariance (state transition) [Pxy]
H = np.zeros(N)         # Innovation covariance (measurement) [Pyy]
R = 1e-2                # observation covariance (sensor noise)
Q = 1e-3                # transition covariance (action uncertainty)
B = np.zeros(N)         # Corrected estimate from last sample (input control) [x_hat]
x_pred = np.zeros(N)    # predicted estimate and covariancefor this sample [B_minus]
beta = np.zeros(N)
# dummy measurement for first sample
sigma = 0.1
z = np.random.normal(mu, sigma, size = N)


def initialize(context):
    '''
    Initialize stocks and signal processing algorithm; schedule function
    '''
    # stocks for use -- #3 & 4 energy stocks above (Excelon Energy & Consolidated Edison)
    global energy
    context.stock1 = energy['EXC']
    context.stock2 = energy['ED']
    set_benchmark(context.stock1)

    # initial market position
    context.pos = None

    # run trading algorithm at specified frequency
    global algo_freq
    for t in range(30, int(7*60), algo_freq):
        schedule_function(sig_proc_trade_algo,
                          date_rules.every_day(),
                          time_rules.market_open(minutes=t))

def sig_proc_trade_algo(context, data):
    '''
    Execute trading strategy based on signal processing technique
    '''
    # get filter parameters
    global N, mu, K, A, H, R, Q, B, x_pred, beta

    #process data
    x = np.asarray(data.history(context.stock1, "price", N, "1m")).reshape((1, N))
    y = data.current(context.stock2, 'price')
    yhat = x.dot(beta)  # initial prediction
    e = y - yhat        # estimator
    # reset yhat and e when inf or nan returned
    if yhat == np.inf or yhat == np.nan or e == np.inf or e == np.nan:
        yhat = y
        e = 0

    bands = np.sqrt(Q) * np.array([-1, 1])  # signal bands

    # record bands and value
    record(Estimator = e, LBound = bands[0], UBound = bands[1])

    # intial values
    B[0] = yhat
    A[0] = 1.0

    for k in range(1,N):
        # update time
        x_pred[k] = B[k-1]
        H[k] = A[k-1] + Q

        # measurement update (Steady State Covariance)
        K[k] = H[k] / (A[k] + R)
        B[k] = x_pred[k] + K[k] * (z[k] - x_pred[k])
        A[k] = (1 - K[k]) * H[k]

    beta = beta + K.flatten()

    # exectue trades
    if context.pos is None: # open positions
        # get number of shares, allocating cash_share between pair
        global cash_share
        cash_to_trade = context.portfolio.cash * cash_share
        shares1 = np.floor((cash_to_trade / 2) / data.current(context.stock1, 'price'))
        shares2 = np.floor((cash_to_trade / 2) / data.current(context.stock2, 'price'))
        if e < bands[0]:
            log.info('Opening long of %d shares of %s' %(shares2, context.stock2.symbol))
            order(context.stock2, shares2)
            order(context.stock1, -shares1 * beta[0])
            context.pos = 'long'
        elif e > bands[1]:
            log.info('Opening short of %d shares of %s' %(shares2, context.stock2.symbol))
            order(context.stock2, -shares2)
            order(context.stock1, shares1 * beta[0])
            context.pos = 'short'
    else:   # close positions
        if context.pos == 'long' and e > bands[0]:
            log.info('Closing long on %s' %(context.stock2.symbol))
            order_target(context.stock1, 0)
            order_target(context.stock2, 0)
            context.pos = None
        elif context.pos == 'short' and e < bands[1]:
            log.info('Closing short on %s' %(context.stock2.symbol))
            order_target(context.stock1, 0)
            order_target(context.stock2, 0)
            context.pos = None
