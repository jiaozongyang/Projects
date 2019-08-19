import numpy as np
import pandas as pd
import xlrd
from datetime import datetime
import matplotlib.pyplot as plt
from utilis import *

df = pd.read_excel('数据集.xlsx', sheet_name = 0)
df2 = pd.read_excel('数据集.xlsx', sheet_name = 1, index_col=0)

def DataProcessing(df, df2):
    
    date_tuple = [xlrd.xldate_as_tuple(i, datemode=0) for i in df.iloc[:,0].values]
    date = [datetime(*tup).strftime('%Y-%m-%d') for tup in date_tuple]
    df.iloc[:,0]  =  date
    df.columns = ['Date','Stock','Bond']
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.drop(index = pd.to_datetime("2019-08-01"), inplace=True)
    
    stock_class = list(df2.columns[:6])
    bond_class = list(df2.columns[6:])
    
    df2 = df2.pct_change().dropna()
    df2_recent = df2.loc['2010-01-01':,]
    
    df2_volatility = df2.rolling(12).std().dropna()
    df2_volatility = df2_volatility.loc['2009-12-31':,]
    
    temp = 1/df2_volatility
    temp['Stock'] = temp.loc[:,stock_class].sum(axis=1)
    temp['Bond'] = temp.loc[:,bond_class].sum(axis=1)
    stock_weight = temp.loc[:,stock_class].values/temp['Stock'].values.reshape((116,1))
    bond_weight =  temp.loc[:,bond_class].values/temp['Bond'].values.reshape((116,1))
    weight = pd.DataFrame(np.hstack((stock_weight, bond_weight)),columns=df2.columns, index=temp.index)
    
    return df, df2_recent, weight
    
    
df, df2, weight = DataProcessing(df, df2)
    
    
class Strategy5(Strategy):
    
    def __init__(self, class_returns, subclass_returns):
        
        self.class_returns = class_returns
        self.subclass_returns = subclass_returns
        
    def generate_signals(self):
        signals = pd.DataFrame(index=self.class_returns.index).fillna(0)
        signals['Stock'] = np.where(df['Stock']>0, df['Stock']/(df['Stock']+df['Bond']),0)
        signals['Bond'] = np.where(df['Stock']>0, df['Bond']/(df['Stock']+df['Bond']), 1)
        return signals    
    
    
class Portfolio5(Portfolio):
    
    def __init__(self, class_returns, subclass_returns, stock_subclass, bond_subclass, signals, weight, initial_capital=10000.0):
        
        self.class_returns = class_returns  # dataframe df
        self.subclass_returns = subclass_returns  # dataframe df2
        self.stock_subclass = stock_subclass    # list stock subclasses
        self.bond_subclass = bond_subclass    # list bond subclasses
        self.signals = signals
        self.initial_capital = float(initial_capital)    
        self.weight = weight
        
        
    def backtest_portfolio(self):
        positions_matrix = np.zeros((self.subclass_returns.shape[0],self.subclass_returns.shape[1]))
        portfolio_matrix = np.zeros((self.subclass_returns.shape[0],self.subclass_returns.shape[1]))
        profits = []
        start_capital = np.zeros((self.subclass_returns.shape[0],))
        start_capital[:12] = self.initial_capital/12
        df_annual_cumulative = (1+self.subclass_returns).rolling(12).apply(np.prod,raw=True)-1
        
        positions_matrix = np.zeros((self.subclass_returns.shape[0],self.subclass_returns.shape[1]))
        portfolio_matrix = np.zeros((self.subclass_returns.shape[0],self.subclass_returns.shape[1]))
        
        
        for i in range(positions_matrix.shape[0]):
    
            positions_matrix[i, :6] = start_capital[i] * self.signals.iloc[i,0] * self.weight.iloc[i,:6]
            positions_matrix[i, 6:] = start_capital[i] * self.signals.iloc[i,1] * self.weight.iloc[i,6:]
            
            if i + 12 <= 114:
                portfolio_matrix[i,:] = positions_matrix[i,:] * df_annual_cumulative.iloc[i+11,:]
                profit = portfolio_matrix[i,:].sum()
                profits.append(profit)
                start_capital[i+12] = start_capital[i] + profit
            else:
                continue
                
        portfolio = pd.DataFrame(portfolio_matrix, index=self.subclass_returns.index, columns=self.subclass_returns.columns)
        position = pd.DataFrame(positions_matrix, index=self.subclass_returns.index, columns=self.subclass_returns.columns)

        portfolio['Start_Capital'] = pd.Series(start_capital).rolling(12).sum().values
        portfolio.loc[:'2018-07-31','Profit'] = profits/(start_capital[:103])
        effective_return = (portfolio.loc['2019-07-18','Start_Capital']/self.initial_capital)**(12/103)-1
        annualized_volatility = np.std(portfolio['Profit'])
        sharp_ratio = np.mean(portfolio['Profit'])/np.std(portfolio['Profit'])
        
        equity = portfolio['Start_Capital'].dropna()
        i = equity.index[np.argmax(np.maximum.accumulate(equity.values) - equity.values)]
        j = equity.index[np.argmax(equity[:i].values)]
        max_drawdown = (equity[j] - equity[i])/equity[j]
        calmar_ratio = effective_return/max_drawdown
        
        cache = {'effective_annual_return': effective_return,
                 'sharp_ratio':sharp_ratio,
                 'volatility': annualized_volatility,
                 'maximum_drawdown': max_drawdown,
                 'calmar_ratio': calmar_ratio,
                 'drawdown_start': j,
                 'drawdown_end': i}
        
        # position weight calculation
        position['StockHolding'] = position.iloc[:,:6].sum(axis=1)
        position['BondHolding'] = position.iloc[:,6:11].sum(axis=1)
        position['AnnualizedStockHolding'] = position['StockHolding'].rolling(12).sum()
        position['AnnualizedBondHolding'] = position['BondHolding'].rolling(12).sum()
        position['StockWeight'] = position['AnnualizedStockHolding']/(position['AnnualizedStockHolding']+position['AnnualizedBondHolding'])
        position['BondWeight'] = position['AnnualizedBondHolding']/(position['AnnualizedStockHolding']+position['AnnualizedBondHolding'])
    
        return portfolio, position, cache
        
        
if __name__ == "__main__":
    class_returns = df
    subclass_returns = df2
    stock_subclass = list(df2.columns[:6])
    bond_subclass = list(df2.columns[6:])
    initial_capital = 10000.0
    StrategyFive = Strategy5(df, df2)
    signals = StrategyFive.generate_signals()

    PortfolioFive = Portfolio5(class_returns, subclass_returns, stock_subclass, bond_subclass, signals, weight, initial_capital)
    portfolio5, position5, cache5 = PortfolioFive.backtest_portfolio()  
        
        
        
        
        
        
        
    
    
    