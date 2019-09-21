import statsmodels.api as sm
from statsmodels.tsa import holtwinters, arima_model
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


def forecast(ts, alpha=None, beta=None, gamma=None, initial_level=None, initial_slope=None, initial_seasons=None, trend=None, seasonal=None, seasonal_periods=None, debug=False):
    """Forecasting using exponential smoothing

    Usage:
        forecast(ts=ts)
        """
    smoothing_level = None
    smoothing_slope = None
    smoothing_seasonal = None
    try:
        if alpha and (0 <= alpha <= 1):
            smoothing_level = alpha
        if beta and (0 <= beta <= 1):
            smoothing_slope = beta
            # if beta is set but trend is None, trend is set equal to add
            if smoothing_slope and not trend:
                trend = 'add'
        if gamma and (0 <= gamma <= 1):
            smoothing_seasonal = gamma
            if smoothing_seasonal and not seasonal:
                seasonal = 'mul'

    except TypeError:
        print('ERRORE: alpha must be between 0 and 1')
        return np.nan

    # if trend:
    #     trend = 'add'
    # else:
    #     trend = None

    # if seasonal:
    #     seasonal = 'add'
    # else:
    #     seasonal = None

    if seasonal and not seasonal_periods:
        print('ERROR: Please, provide the number of periods in a complete seasonal cycle')
        return np.nan

    model = holtwinters.ExponentialSmoothing(endog=pd.DataFrame(data=ts),
                                             trend=trend,
                                             seasonal=seasonal,
                                             seasonal_periods=seasonal_periods).fit(smoothing_level=smoothing_level,
                                                                                    smoothing_slope=smoothing_slope,
                                                                                    smoothing_seasonal=smoothing_seasonal,
                                                                                    initial_level=initial_level,
                                                                                    initial_slope=initial_slope)

    model.predict(start=1, end=len(ts))

    if debug:
        print(model.model.params)
    # print('TS')
    # print(ts)
    # print('FORECAST')
    # print(model.fittedfcast)
    return model.fittedfcast


def arima(ts):
    model = arima_model.ARIMA(ts, ())


def mape(ts, forecast):
    """ Computes the Mean Average Percentage Error (MAPE)
        Usage:
            mape(ts=time_series, forecast=forecast)

        Parameters:
            ts (list): A time series
            forecast (list): A forecast (computed with the forecast() function)

        Returns:
            list: MAPE
    """
    err = []
    for d, f in zip(ts, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(abs(r) / d)
        if d < 0:
            # print(d, f)
            pass
    return np.mean(err) * 100


def me(ts, forecast):
    """ Computes the Mean Error (ME)
        Usage:
            me(ts=time_series, forecast=forecast)

        Parameters:
            ts (list): A time series
            forecast (list): A forecast (computed with the forecast() function)

        Returns:
            list: ME
    """

    err = []
    for d, f in zip(ts, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(r)
    # print(err)
    return np.mean(err)


def mse(ts, forecast):
    """ Computes the Mean Squared Error (ME)
        Usage:
            mse(ts=time_series, forecast=forecast)

        Parameters:
            ts (list): A time series
            forecast (list): A forecast (computed with the forecast() function)

        Returns:
            list: MSE
    """

    err = []
    for d, f in zip(ts, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(r ** 2)

    return np.mean(err)


def mae(ts, forecast):
    """ Computes the Mean Absolute Error (MAE) (also know as Mean Absolute Deviation (MAD))
        Usage:
            mae(ts=time_series, forecast=forecast)

        Parameters:
            ts (list): A time series
            forecast (list): A forecast (computed with the forecast() function)

        Returns:
            list: MAE (or MAD)
    """

    err = []
    for d, f in zip(ts, forecast):
        r = d - f
        if not math.isnan(r):
            err.append(abs(r))

    return np.mean(err)


def kpi(ts, forecast, round_to=1):
    """ Computes all the errors
        Usage:
            kpi(ts=time_series, forecast=forecast, round_to=round_to)

        Parameters:
            ts (list): A time series
            forecast (list): A forecast (computed with the forecast() function)
            round_to (int >= 0): Number of digits after comma

        Returns:
            Prints the errors on screen
    """
    if round_to < 0:
        print('Please, provide a rounding value >= 0 (default is 1)')
        return
    print('ME: ', round(me(ts, forecast), round_to))
    print('MAE: ', round(mae(ts, forecast), round_to))
    print('MAPE: ', round(mape(ts, forecast), round_to), '%')
    print('MSE: ', round(mse(ts, forecast), round_to))
