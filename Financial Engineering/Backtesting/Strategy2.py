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
    
    df2_percent = df2.pct_change()
    df2_percent = df2_percent.loc['2010-01-01':,]
    
    return df, df2_percent

df, df2  = DataProcessing(df, df2)



class Strategy2(Strategy):
    
    def __init__(self, class_returns, subclass_returns):
        
        self.class_returns = class_returns
        self.subclass_returns = subclass_returns
        
    def generate_signals(self):
        signals = pd.DataFrame(index=self.class_returns.index).fillna(0)
        signals['Stock'] = np.where(self.class_returns['Stock']>0, self.class_returns['Stock']/(self.class_returns['Stock']+ self.class_returns['Bond']), 0)
        signals['Bond'] = np.where(self.class_returns['Stock']>0, self.class_returns['Bond']/(self.class_returns['Stock']+ self.class_returns['Bond']), 1)
        
        return signals
    
    
    
    
    
class Portfolio2(Portfolio):
    
    def __init__(self, class_returns, subclass_returns, stock_subclass, bond_subclass, signals, initial_capital=10000.0):
        
        self.class_returns = class_returns  # dataframe df
        self.stock_subclass = stock_subclass    # list stock subclasses
        self.bond_subclass = bond_subclass    # list bond subclasses
        self.subclass_returns = subclass_returns  # dataframe df2
        self.signals = signals
        self.initial_capital = float(initial_capital)
        
    
    def backtest_portfolio(self):
        positions_matrix = np.zeros((self.subclass_returns.shape[0],self.subclass_returns.shape[1]))
        portfolio_matrix = np.zeros((self.subclass_returns.shape[0],self.subclass_returns.shape[1]))
        capital = self.initial_capital
        profits = []
        
        for i in range(positions_matrix.shape[0]):
            
            positions_matrix[i, :6] = capital * signals.iloc[i,0]/len(self.stock_subclass)
            positions_matrix[i, 6:] = capital * signals.iloc[i,1]/len(self.bond_subclass)

            portfolio_matrix[i,:] = positions_matrix[i,:] * self.subclass_returns.iloc[i,:]
            profit = portfolio_matrix[i,:].sum()
            profits.append(profit)

            capital += profit
            
            
        portfolio = pd.DataFrame(portfolio_matrix, index=self.subclass_returns.index, columns=self.subclass_returns.columns)
        portfolio['Profit'] = profits
        portfolio['Capital'] = portfolio['Profit'].cumsum() + self.initial_capital
        portfolio['Start_Capital'] = portfolio['Capital'].shift(1)
        portfolio['Start_Capital'].fillna(self.initial_capital, inplace=True)
        portfolio['Annualized_Return'] = (portfolio['Profit']/portfolio['Start_Capital'] + 1)**12 - 1
        # annualized return of whole period
        effective_return = (portfolio.loc['2019-07-18','Capital']/self.initial_capital)**(12/115)-1
        
         #calculate sharp ratio
        annualized_volatility = np.std(portfolio['Profit']/portfolio['Start_Capital']) * np.sqrt(12)
        sharp_ratio = portfolio['Annualized_Return'].mean() /annualized_volatility
        
        #calculate maximum drawdown
        equity = portfolio['Start_Capital'].dropna()
        i = equity.index[np.argmax(np.maximum.accumulate(equity.values) - equity.values)]
        j = equity.index[np.argmax(equity[:i].values)]
        max_drawdown = (equity[j] - equity[i])/equity[j]
        calmar_ratio = effective_return/max_drawdown
        
   
        cache = {'effective_annual_return':effective_return,
                'sharp_ratio':sharp_ratio,
                'volatility': annualized_volatility,
                'maximum_drawdown': max_drawdown,
                'calmar_ratio':calmar_ratio,
                'drawdown_start': j,
                'drawdown_end': i}
    
        return portfolio, cache
        
    
    
    
if __name__ == "__main__":
    class_returns = df
    subclass_returns = df2
    stock_subclass = list(df2.columns[:6])
    bond_subclass = list(df2.columns[6:])
    initial_capital = 10000.0
    StrategyTwo = Strategy2(df, df2)
    signals = StrategyTwo.generate_signals()
    
    PortfolioTwo = Portfolio2(class_returns, subclass_returns, stock_subclass, bond_subclass, signals, initial_capital)
    
    portfolio2, cache2 = PortfolioTwo.backtest_portfolio()