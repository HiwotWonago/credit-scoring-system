#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('C:\\Users\\Hiwi\\Documents\\week5\\data.csv')


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols].hist(bins=30, figsize=(15, 10))


# In[9]:


categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    sns.countplot(y=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()


# In[ ]:


corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')


# In[ ]:


missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_percent})
missing_df[missing_df['Missing Count'] > 0]


# In[ ]:


for col in numerical_cols:
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[ ]:


#import os
#os.chdir('C:/Users/Hiwi/Documents/week5/') 


# In[ ]:


#!git add .
#!git commit -m "Initial commit: EDA and feature engineering"


# In[ ]:


#!git remote add origin https://github.com/HiwotWonago/credit-scoring-system.git
#!git branch -M main
#!git push -u origin main


# In[1]:


from pipelines import build_transaction_pipeline
from feature_engineering import TransactionTimeFeatures

# Build pipeline
feature_pipeline = build_feature_pipeline()

# Process data
processed_data = feature_pipeline.fit_transform(df)

# Convert to DataFrame (retain column names)
processed_df = pd.DataFrame(
    processed_data,
    columns=feature_pipeline.get_feature_names_out()
)


# In[10]:


from sklearn.base import BaseEstimator, TransformerMixin

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


# In[13]:


# src/pipelines.py

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from feature_engineering import TransactionTimeFeatures

# ================================
# Define column groups
# ================================

CATEGORICAL_COLS = [
    'CurrencyCode', 'ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy'
]

NUMERICAL_COLS = [
    'Amount', 'Value'
]

DATETIME_COL = 'TransactionStartTime'

# ================================
# Pipelines for sub-transforms
# ================================

# Pipeline for categorical features
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Pipeline for numerical features
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# ================================
# Preprocessing ColumnTransformer
# ================================

def build_transaction_pipeline():
    preprocessing = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, NUMERICAL_COLS),
        ('cat', categorical_pipeline, CATEGORICAL_COLS)
    ])

    # Final pipeline with datetime features first
    full_pipeline = Pipeline(steps=[
        ('datetime_features', TransactionTimeFeatures(datetime_col=DATETIME_COL)),
        ('preprocessing', preprocessing)
    ])

    return full_pipeline


# In[ ]:




