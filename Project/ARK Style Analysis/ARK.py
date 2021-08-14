import pandas as pd
import numpy as np
import pandas_datareader as pdr
import pandas_datareader.data as web
from pandas_datareader.famafrench import get_available_datasets
import statsmodels.api as sm
import os
from scipy.optimize import minimize
from scipy.stats import norm
import datetime
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import shap
from xgboost import XGBRegressor
from functools import reduce
from box import Box
import yaml
from typing import List

# ARKK Innovation
# ARKQ Autonomous & Robotics
# ARKW Internet
# ARKG Genomic
# ARKF Fintech

with open("config.yaml","r") as ymlfile:
    cfg = Box(yaml.safe_load(ymlfile))
    
START_DATE = "2010-01-01"
END_DATE = "2021-06-30"

TIINGO_LIST = ['TSLA',"AIA",'ARKW','ARKK','ARKQ',
               'ARKG','GBTC','BTCUSD','SPYG','MTUM']

VARS = ['Mkt-RF','NASDAQ_ret-RF','AIA_ret-RF','HML','Mom',
        'SPYG_ret','TSLA_ret','ARKW_ret','ARKK_ret',
        'ARKQ_ret','ARKG_ret','GBTC_ret']

COLORS = px.colors.qualitative.Dark24

d = {
    'ARKW_ret': [
            ['NASDAQ_ret-RF','AIA_ret-RF','HML', 'Mom','TSLA_ret','GBTC_ret'],
            ['NASDAQ_ret-RF','AIA_ret-RF','TSLA_ret','GBTC_ret'],
            ['Mkt-RF','AIA_ret-RF','HML', 'Mom','TSLA_ret','GBTC_ret']
    ],
    'ARKK_ret':[
            ['NASDAQ_ret-RF','AIA_ret-RF','HML', 'Mom','TSLA_ret'],
            ['NASDAQ_ret-RF','AIA_ret-RF','TSLA_ret'],
            ['Mkt-RF','AIA_ret-RF','HML', 'Mom','TSLA_ret']
    ],
     'ARKQ_ret':[
            ['NASDAQ_ret-RF','AIA_ret-RF','HML', 'Mom','TSLA_ret'],
            ['NASDAQ_ret-RF','AIA_ret-RF','TSLA_ret'],
            ['Mkt-RF','AIA_ret-RF','HML', 'Mom','TSLA_ret']
    ],
    'ARKG_ret':[
            ['NASDAQ_ret-RF','HML', 'Mom','covid'],
            ['NASDAQ_ret-RF','HML', 'Mom'],
            ['Mkt-RF','HML', 'Mom','covid'],
            ['Mkt-RF','HML', 'Mom']
    ]
}


def getFamaFrench(freq=''):
    '''
    Please note that there is no weekly momentum data
    start: str (format: YYYY-MM-DD)
    end: str (format: YYYY-MM-DD)
    freq: str (_daily/default)
    '''
    factors_str = f"F-F_Research_Data_Factors{freq}"
    mom_str = f"F-F_Momentum_Factor{freq}"
    
    factors = web.DataReader(factors_str,'famafrench',start = START_DATE, end = END_DATE)
    mom = web.DataReader(mom_str,'famafrench',start = START_DATE, end = END_DATE)
    factors = pd.concat([factors[0]/100, mom[0]/100], axis=1).reset_index()
    factors.rename({'Mom   ':'Mom', 'Date':'date'}, axis=1, inplace=True)
    return factors

def getNASDAQ():
    test = web.DataReader('NASDAQCOM', 'fred', START_DATE, END_DATE)
    test.index = pd.to_datetime(test.index)
    test['month'] = test.index.to_period("M")
    test = test.groupby('month')['NASDAQCOM'].last().reset_index()
    test['NASDAQ_ret'] = test['NASDAQCOM'] / test['NASDAQCOM'].shift() - 1
    return test[['month','NASDAQ_ret']]

def processTiingoData(df):
    var = df['symbol'][0]
    df.set_index('date',inplace=True)
    df.index = pd.to_datetime(df.index)
    df['month'] = df.index.to_period("M")
    df = df.groupby('month')['adjClose'].last().reset_index()
    df[f'{var}_ret'] = df['adjClose'] / df['adjClose'].shift() - 1
    df = df[['month', f'{var}_ret']]
    return df

def getTiingo(tiingo_list = TIINGO_LIST):
    df_list = [pdr.get_data_tiingo(ticker, api_key=cfg.base.api.key, start=START_DATE, end=END_DATE).reset_index() for ticker in tiingo_list]
    df_list = [processTiingoData(df) for df in df_list]
    df = reduce(lambda x, y: pd.merge(x, y, on = 'month', how = 'outer'), df_list)
    df.dropna(how='all',subset=['ARKW_ret','ARKK_ret','ARKQ_ret','ARKG_ret'],axis=0,inplace=True)
    return df
   
    
def mergeData(factors, df, nasdaq):
    df = df.merge(factors,how='left',left_on='month',right_on='date').merge(nasdaq, how='left', on='month')
    df['GBTC_ret'].fillna(df['BTCUSD_ret'],inplace=True)
    df.drop(['BTCUSD_ret','date'], axis=1, inplace=True)
    df['NASDAQ_ret-RF'] = df['NASDAQ_ret'] - df['RF']
    df['AIA_ret-RF'] = df['AIA_ret'] - df['RF']
    
    df.loc[(df['month'] >= "2020-02") & (df['month'] <= "2020-12"),'covid'] = 1
    df['covid'].fillna(0, inplace=True) 
    return df

def _regression(X, y):
    return sm.OLS(y, sm.add_constant(X)).fit()

def regression(df, Xs: List, y_str: str, subset = False):
    if subset:
        df_copy = df.dropna(how='any', axis=0).tail(250)
    else:
        df_copy = df.copy()
    df_copy = df_copy[Xs + [y_str]].dropna(how='any',axis=0)
    
    return _regression(df_copy[Xs], df_copy[y_str])

def prepare(params: List, d):
        result = pd.concat(params, axis=1).T
        result['Fund'] = sum([np.repeat(key, len(d[key])).tolist() for key in d], [])
        #result['Data'] = ['All Data','Recent Data'] * (int(len(result)/2))
        result = result.set_index(['Fund']).sort_index()
        return result

def getModel(df, d):
    params = []
    pvalues = []
    for y_var, xs_lists in d.items():
        for xs_list in xs_lists:
            model_all = regression(df, xs_list, y_var)
            params.append(model_all.params)
            pvalues.append(model_all.pvalues)            
    
    params_df = prepare(params, d)
    pvalues_df = prepare(pvalues, d)
    return params_df, pvalues_df

def style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 0.8),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    solution = minimize(portfolio_tracking_error, init_guess,
                       args=(dependent_variable, explanatory_variables,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    weights = pd.Series(solution.x, index=explanatory_variables.columns)
    return weights

def portfolio_tracking_error(weights, ref_r, bb_r):
    """
    returns the tracking error between the reference returns
    and a portfolio of building block returns held with given weights
    """
    def tracking_error(r_a, r_b):
        """
        Returns the Tracking Error between the two return series
        """
        return np.sqrt(((r_a - r_b)**2).sum())
    return tracking_error(ref_r, (weights*bb_r).sum(axis=1))

def annualize_rets(r, periods_per_year=12):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year=12):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    """
    r = r.dropna(axis=0)
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
        
        
def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """
    r = r.dropna(axis=0)
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")

def var_gaussian(r, level=5, modified=False):
    """
    Returns the Parametric Gauusian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z score assuming it was Gaussian
    r = r.dropna(axis=0)
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z +
                (z**2 - 1)*s/6 +
                (z**3 -3*z)*(k-3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )

    return -(r.mean() + z*r.std(ddof=0))


def sharpe_ratio(r, riskfree_rate=0.02, periods_per_year=12):
    """
    Computes the annualized sharpe ratio of a set of returns
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def maxDrawdown(return_series):
    """ Takes a time series of asset returns.
        returns a series with columns for the wealth,
               a dictionary with max drawdown,
               point of time at max and min
    """
    wealth = (1 + return_series).cumprod()
    previous_peaks = wealth.cummax()
    drawdowns = (wealth - previous_peaks) / previous_peaks

    #cache = {'MaxDrawdown': drawdowns.min(),
    #         'down': drawdowns.idxmin(),
    #         'up': previous_peaks[:drawdowns.idxmin()].idxmax()}
    return drawdowns.min()

def marketTiming(df, y, x):
    df_copy = df.copy()
    df_copy['y'] = df_copy[y] - df_copy['RF']
    df_copy['x1'] = df_copy[x] - df_copy['RF']
    df_copy['dummy'] = (df_copy[x] > df_copy['RF']) * 1.0
    df_copy['x2'] = df_copy['x1'] * df_copy['dummy']
    df_copy.dropna(how='any', subset=['y','x1','x2'], inplace=True)
    model = _regression(df_copy[['x1','x2']], df_copy['y'])
    return model


def rollingRegress(df, Xs_list, y_str, num_record=24):
    dates=[]
    parameters=[]
    df_copy = df[Xs_list + [y_str, 'month']].dropna(how='any',axis=0).reset_index()
    for i in range(num_record, len(df_copy)-1):
        X = df_copy.loc[i-num_record:i, Xs_list]
        y = df_copy.loc[i-num_record:i, y_str]
        date = df_copy.loc[i, 'month']
        model = _regression(X, y)
        param = model.params
        dates.append(date)
        parameters.append(param)
        
    result = pd.concat(parameters, axis=1).T
    result.index = dates
    result = result.reset_index()
    
    return result


def coefficientsPlot(result_df, Xs_list, y_str, names):
    colors = ['lightblue', 'darkblue', 'firebrick','black', 'magenta', 'green']
    fig = make_subplots(rows=2, cols=1)
    result_df['index'] = result_df['index'].apply(lambda x: x.to_timestamp())
    for i in range(len(Xs_list)):
        fig.add_trace(
            go.Scatter(x=result_df["index"], y=result_df[Xs_list[i]], 
                                 name=names[i],line = dict(color=colors[i], width=2)),
        row = 1, col = 1)

    fig.add_trace(
            go.Scatter(x=result_df["index"], y=result_df["const"], 
                             name="Alpha",line = dict(color='royalblue', width=2)),
        row = 2, col = 1)

    fig.update_layout(height=800, width=800, title_text=f"{y_str} Coefficients of Rolling Regression")
    fig.show()
    
    
def xgboostModel(df, Xs_list, y_str, plot=True):
    df_copy = df[Xs_list + [y_str, 'date']].dropna(how='any',axis=0)
    X_train = df_copy[Xs_list]
    y_train = df_copy[y_str]
    xgb = XGBRegressor(n_estimators=500, max_depth = 5, 
                       learning_rate=0.1, n_jobs=4, objective = 'reg:squarederror').\
            fit(X_train, y_train, eval_set=[(X_train, y_train)],eval_metric='rmse',verbose=False)
    if plot:
        y_pred = pd.DataFrame(xgb.predict(X_train), columns=[f"{y_str}_pred"])
        df_res = pd.concat([y_pred, y_train], axis = 1)
        (df_res + 1).cumprod().plot()
    return xgb, X_train, y_train