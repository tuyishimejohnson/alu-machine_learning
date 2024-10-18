#!/usr/bin/env python3

# Import pandas
import pandas as pd

# Import '2-from_file'

from_file = __import__('2-from_file').from_file

df = from_file('pipeline/pandas/datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
df = df[['Datetime', 'Close']]

print(df.tail())
