from abc import ABCMeta, abstractmethod

class Strategy(object):
    __metaclass__ = ABCMeta
    @abstractmethod
    def generate_signals(self):
        raise NotImplementedError("Need to implement generate_positions()!")
        
        
class Portfolio(object):
    __metaclass__ = ABCMeta
        
    @abstractmethod
    def backtest_portfolio(self):
        raise NotImplementedError("Need to implement backtest_portfolio()!")
        
        

       