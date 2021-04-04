import pandas_datareader.data as reader
import pandas as pd
import datetime as dt
import statsmodels.api as sm

end = dt.date(2020, 6, 30)
start = dt.date(end.year - 5, end.month, end.day)

funds = ['FDGRX']

fundret = reader.get_data_yahoo(funds, start, end)['Adj Close'].pct_change()

fundsret_mtl = fundret.resample('M').agg(lambda x: (x + 1).prod() - 1)  # cumulative monthly

fundsret_mtl = fundsret_mtl[1:]  # remove zero return at the start

## now for the factor downloads

factors = reader.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0]
factors = factors[1:]

fundsret_mtl.index = factors.index  # make date formats the same so we can merge

merge = pd.merge(fundsret_mtl, factors, on='Date')

merge[['Mkt-RF', 'SMB', 'HML', 'RF']] = merge[['Mkt-RF', 'SMB', 'HML', 'RF']] / 100

merge['FDGRX-RF'] = merge.FDGRX - merge.RF  # LHS of FF 3 Factor model
print(merge.head(2))

y = merge['FDGRX-RF']
X = merge[['Mkt-RF', 'SMB', 'HML']]

X_sm = sm.add_constant(X)
model = sm.OLS(y, X_sm)
results = model.fit()
summary = results.summary()
print(summary)
