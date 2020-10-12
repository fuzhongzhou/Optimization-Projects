import wrds
import numpy as np
import pandas as pd

db = wrds.Connection()

stocksql = "select date, cusip, prc as price, cfacpr as adjfactor, vol \
            from crsp.dsf where date between '2015-01-01' and '2018-12-31' \
            and vol is not NULL \
            and cusip in (select cusip from crsp.dsf where date='2015-01-02' order by vol DESC limit 200)"

stock_dt = db.raw_sql(stocksql)
stock_dt['adjprice'] = stock_dt.price * stock_dt.adjfactor
stock_dt.set_index(['date', 'cusip'], inplace=True)
stock_dt.sort_index(inplace=True)
stock_dt['ret'] = stock_dt.price.groupby('cusip').pct_change()
stock_dt.dropna(inplace=True)

stock_dt.to_csv('stock.csv')

def read_factor(filename):
    dt = pd.read_csv(filename)
    dt = dt[(dt.date > 20150103) & (dt.date <= 20181231)]
    dt.date = pd.to_datetime(dt.date, format='%Y%m%d')
    dt.set_index('date', inplace=True)
    return dt

fama_dt = read_factor('fama.csv')
STREV_dt = read_factor('ShrtermReversal.csv')
LTREV_dt = read_factor('LongtermReversal.csv')
industry_dt = read_factor('industry.csv')

momentum_sql = "select date, umd as momentum \
       from ff.factors_daily where date between '2015-01-05' and '2018-12-31'"
mom = db.raw_sql(momentum_sql)

mom.set_index('date', inplace=True)
fama_dt['MOM'] = mom.momentum

factors = fama_dt.join([STREV_dt, LTREV_dt, industry_dt], how='inner')
factors.to_csv('factors.csv')