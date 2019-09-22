import statsmodels.api as sm
from statsmodels.tsa import holtwinters
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.tsatools import detrend as ts_detrend
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import math
import warnings

print('WARNING: This software is designed and provided for educational purpose only')
print('You may refer to https://otexts.com/fpp2/ for theory')


def data(series_id=None):
    """ Returns a time series from the archive
        Usage:
            data(series_id=series_id)

        Parameters:
            series_id (int): id of the time series

        Returns:
            list: time_series"""
    warnings.filterwarnings("ignore")
    if not series_id:
        print('Please, provide a series id')
        return

    if series_id == 1:
        dataset = sm.datasets.copper.load()
        return [each[0] for each in dataset.data]
    elif series_id == 2:
        df = pd.read_csv('series2.csv')
        return [each[0] for each in df.values]
    elif series_id == 3:
        df = pd.read_csv('series3.csv')
        return [each[0] for each in df.values]
    elif series_id == 4:
        df = pd.read_csv('series4.csv')
        return [each[0] for each in df.values]
    elif series_id == 5:
        df = pd.read_csv('series5.csv')
        return [each[0] for each in df.values]
    else:
        print(f'ERROR: series_id {series_id} not found')


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


def acf_plot(ts):
    """ Plots the ACF

        Usage:
            acf_plot(ts=ts)

        Parameters:
            ts (list): A time series

        Returns:
            An ACF plot
    """
    if not isinstance(ts, pd.DataFrame):
        ts = pd.DataFrame(data=ts)
    plot_acf(ts)
    plt.show()


def detrend(ts):
    """ Detrend a time series

        Usage:
            detrend(ts=ts)

        Parameters:
            ts (list): A time series

        Returns:
            list: The detrended time series
        """
    det = ts_detrend(np.array(ts), order=1)
    return det.tolist()


def decompose(ts, model='add', seasonal_periods=12):
    """ Decompose a time series accordinf to an additive (model = 'add') or multiplicative (model = 'mul') model

        Usage:
            decompose(ts=ts, model=model)

        Parameters:
            ts (list): A time series
            model (str): 'add' for an additive model decomposition, 'mul' for a multiplicative model
            seasonal_periods (int): Periods in a season

        Returns:
            Plot the decomposes time series
    """
    if model == 'add':
        model = 'additive'
    elif model == 'mul':
        model = 'multiplicative'
    else:
        print(f'ERROR: decompostion model {model} not available')
        return
    return seasonal_decompose(ts, model=model, freq=seasonal_periods).plot()


def mean(ts):
    """ Calculate tne mean value of the time series and uses it as a forecast

        Usage:
            mean(ts=ts)

        Parameters:
            ts (list): A time series

        Returns:
            list: forecast as mean value
    """
    mv = np.mean(ts)
    return [mv for _ in range(len(ts))]


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


def seasonal_naive(ts, seasonal_periods):
    """ Seasonal naive forecasting: each forecast is equal to the last observed value from the same season of the year (e.g., the same month of the previous year)"""
    print('NOT IMPLEMENTED YET')


def forecast(ts, alpha=None, beta=None, gamma=None, initial_level=None, initial_slope=None, initial_seasons=None, trend=None, seasonal=None, seasonal_periods=None, debug=False, use_boxcox=False):
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
                                                                                    initial_slope=initial_slope,
                                                                                    use_boxcox=use_boxcox)

    model.predict(start=1, end=len(ts))

    if debug:
        print(model.model.params)
    # print('TS')
    # print(ts)
    # print('FORECAST')
    # print(model.fittedfcast)
    return model.fittedfcast


def fit_arima(ts, p_values=None, d_values=None, q_values=None):
    """ Fits an ARIMA models using a grid search over p, q and d

        Usage:
            fit_arima(ts=ts, p_values=p_values, d_values=d_values, q_values=q_values)

        Parameters:
            ts (list): A time series
            p_values (list of int): The number of lag observations included in the model, also called the lag order.
            d_values (list of int): The number of times that the raw observations are differenced, also called the degree of differencing.
            q_values (list of int): The size of the moving average window, also called the order of moving average.
    """
    if not p_values:
        p_values = list(range(2))
    if not d_values:
        d_values = list(range(2))
    if not q_values:
        q_values = list(range(2))
    return evaluate_models(ts, p_values, d_values, q_values)


def arima(ts=None, p=None, d=None, q=None):
    """ Forecasting using an ARIMA model of order=(p, d, q)"""

    if not ts:
        print('ERROR: please provide a time series')
        return
    if p is None or d is None or q is None:
        print('ERROR: please provide the values for p, d and q')
        return
    model = ARIMA(ts, order=(p, d, q))
    model_fit = model.fit(disp=0)
    model_fit.predict(start=1, end=len(ts))

    return model_fit.fittedvalues


# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    error = mse(test, predictions)
    return error


def evaluate_models(dataset, p_values, d_values, q_values):
    """ Evaluates combinations of p, d and q values for an ARIMA model"""

    warnings.filterwarnings("ignore")
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order, mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


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


def fb_forecast(ts, periods=1):

    warnings.filterwarnings("ignore")
    # Add timeindex to ts
    # Assume daily observations
    date_rng = pd.date_range(start='1/1/2018', periods=len(ts), freq='D')
    values = []
    for d, v in zip(date_rng, ts):
        values.append([d.date(), v])

    # Create the dataframe
    df = pd.DataFrame(data=values, columns=['ds', 'y'])

    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    # return only the values
    return forecast['yhat'].values
