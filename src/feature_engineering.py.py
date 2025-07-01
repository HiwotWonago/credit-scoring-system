#!/usr/bin/env python
# coding: utf-8

# In[4]:


# src/feature_engineering.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# In[5]:


df = pd.read_csv('C:\\Users\\Hiwi\\Documents\\week5\\data.csv')


# In[6]:


class AggregateTransactionFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='CustomerId', amount_col='Amount'):
        self.id_col = id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.id_col)[self.amount_col].agg(
            total_amount='sum',
            avg_amount='mean',
            transaction_count='count',
            std_amount='std'
        ).reset_index()
        return agg

class TransactionTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X


# In[ ]:




