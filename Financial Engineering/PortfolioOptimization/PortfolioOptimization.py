import numpy as np
import pandas as pd
import quandl

quandl.ApiConfig.api_key = 'Enter your API Key'


def GetData(stockList, start, end):
    '''
    Take a list of stock codes, the start
    and the end of analysis period as inputs and return
    a data frame ready for analysis
    '''
    raw_data = quandl.get_table('WIKI/PRICES', ticker=stockList,
                                qopts={'columns': ['ticker', 'date', 'adj_close']},
                                date={'gte': start, 'lte': end},
                                paginate=True)

    clean_data = raw_data.pivot_table(index='date',
                                      columns='ticker',
                                      values='adj_close')

    # I found sometimes there are NA values in dataframe (eg.AAPL),
    # so I fill them with column mean

    if clean_data.isna().any().any():
        #print('NA values in the dataframe')
        clean_data.fillna(clean_data.mean(), inplace=True)
        return clean_data

    else:
        return clean_data


def MeanVarCalculation(df):
    '''
    Take a dataframe of stock closing prices
    as input and outputs an array of stock mean return
    and covariance matrix
    '''
    n = df.shape[0]  # n: number of rows
    m = df.shape[1]  # m: number of columns
    array1 = np.array(df.iloc[1:, :])
    array2 = np.array(df.iloc[0:n - 1, :])

    ret = array1 / array2 - 1
    retMean = np.nanmean(ret, axis=0)
    covMatrix = np.cov(ret, rowvar=False)

    return retMean, covMatrix


def WeightCalculation(covMatrix):
    cov_inv = np.linalg.inv(covMatrix)
    b = np.ones(len(covMatrix))
    weight = np.dot(cov_inv, b)
    weight = weight / weight.sum()

    return weight

stock_list = ['AAPL','MSFT','WMT']
test_data = GetData(stock_list, '2015-01-01', '2018-12-31')
retMean, covMatrix = MeanVarCalculation(test_data)
weight = WeightCalculation(covMatrix)

for index, name in enumerate(stock_list):
    print("The weight of", name, "in the portfolio is", round(weight[index],2))