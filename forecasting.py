import statsmodels.api as sm
from statsmodels.tsa import holtwinters
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import math


def data(series=None):
    """ Returns a time series from the archive
        Usage:
            data(series=series_id)

        Parameters:
            series_id (int): id of the time series

        Returns:
            list: time_series"""

    if series == 1:
        dataset = sm.datasets.copper.load()
        return [each[0] for each in dataset.data]
    elif series == 2:
        df = pd.read_csv('series2.csv')
        return [each[0] for each in df.values]
    elif series == 3:
        df = pd.read_csv('series3.csv')
        return [each[0] for each in df.values]
    elif series == 4:
        df = pd.read_csv('series4.csv')
        return [each[0] for each in df.values]
    else:
        print('Please, provide a series id')


def plot(**kwargs):
    """ Plots the given time series
        Usage:
            plot(ts_name=ts_values [, ts_name=ts_values...])

        Parameters:
            ts_name (str): Name of the time series (displaied in the legenda). One word, no whitespace
            ts_values (list): Time series values (plotted)

        Returns:
            Plot of the time series

        Example:
            ts1 = data(1)
            ts2 = data(2)
            plot(name_of_ts1=ts1, name_of_ts2=ts2)"""

    plt.figure(figsize=(14, 8))
    for key, arg in kwargs.items():
        plt.plot(arg, label=key)
    plt.legend()
    plt.show()


def ma(ts, num_periods=3):
    """ Return the moving average with a num_periods window

        Usage:
            ma(ts=time_series, num_periods=num_periods)

        Parameters:
            ts (list): A time series
            num_periods (int): Length of the moving average window

        Returns:
            list: moving average
        """
    return [np.nan] + [each[0] for each in pd.DataFrame(data=ts).rolling(num_periods).mean().values]


def naive(ts):
    """ Naive forecasting

        Usage:
            naive(ts=time_series)

        Parameters:
            ts (list): A time series of the time series

        Returns:
            list: naive forecasti for the time series"""

    return [np.nan] + ts


def forecast(data, alpha=None, beta=None, gamma=None, trend=False, seasonal=False, seasonal_periods=None, optimized=False, debug=False):
    # Sanifica input

    smoothing_level = None
    smoothing_slope = None
    try:
        if alpha and (0 <= alpha <= 1):
            smoothing_level = alpha
        if beta and (0 <= beta <= 1):
            smoothing_slope = beta
    except TypeError:
        print('ERRORE: Assicurarsi che i valori siano numerici e compresi tra 0 e 1')
        return np.nan

    if trend:
        trend = 'add'
    else:
        trend = None

    if seasonal:
        seasonal = 'add'
    else:
        seasonal = None

    if seasonal and not seasonal_periods:
        print('ERRORE: con il modello stagionale occorre indicare il numero di stagioni')
        return np.nan

    model = holtwinters.ExponentialSmoothing(endog=pd.DataFrame(data=data),
                                             trend=trend,
                                             seasonal=seasonal,
                                             seasonal_periods=seasonal_periods).fit(smoothing_level=smoothing_level,
                                                                                    smoothing_slope=smoothing_slope,
                                                                                    optimized=optimized)

    model.predict(start=1, end=len(data))

    if debug:
        print(model.model.params)
    return model.fittedfcast


def mape(data, forecast):
    err = []
    for d, f in zip(data, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(abs(r) / d)
        if d < 0:
            print(d, f)
    return np.mean(err)


def me(data, forecast):
    err = []
    for d, f in zip(data, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(r)

    return np.mean(err)


def mse(data, forecast):
    err = []
    for d, f in zip(data, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(r ** 2)

    return np.mean(err)


def mae(data, forecast):
    err = []
    for d, f in zip(data, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(abs(r))

    return np.mean(err)


def kpi(data, forecast):
    print('ME: ', me(data, forecast))
    print('MAE: ', mae(data, forecast))
    print('MAPE: ', mape(data, forecast))
    print('MSE: ', mse(data, forecast))
