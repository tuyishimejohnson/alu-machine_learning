#!/usr/bin/env python3

from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file
df = from_file('pipeline/pandas/datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove Weighted_Price
df = df.drop(columns=['Weighted_Price'])

# Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})

# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')

# Index the data frame on Date
df = df.set_index('Date')

# Fill missing values
df['Close'].fillna(method='ffill', inplace=True)
df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)
df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

# Filter data from 2017 and beyond
df = df[df.index >= '2017-01-01']

# Resample the data to daily intervals and aggregate
df_daily = df.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(14, 7))
plt.plot(df_daily.index, df_daily['Open'], label='Open')
plt.plot(df_daily.index, df_daily['High'], label='High')
plt.plot(df_daily.index, df_daily['Low'], label='Low')
plt.plot(df_daily.index, df_daily['Close'], label='Close')
plt.plot(df_daily.index, df_daily['Volume_(BTC)'], label='Volume (BTC)')
plt.plot(df_daily.index, df_daily['Volume_(Currency)'], label='Volume (Currency)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()