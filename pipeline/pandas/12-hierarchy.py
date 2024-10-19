""" #!/usr/bin/env python3 """

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('pipeline/pandas/datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('pipeline/pandas/datasets/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

df1 = df1.loc[1417411980:1417417980]
df2 = df2.loc[1417411980:1417417980]

# Add keys to the data
df1['key'] = 'coinbase'
df2['key'] = 'bitstamp'

# Reset index to add 'Timestamp' as a column
df1 = df1.reset_index()
df2 = df2.reset_index()


df1 = df1.set_index(['Timestamp', 'key'])
df2 = df2.set_index(['Timestamp', 'key'])

# Concatenate the dataframes
df = pd.concat([df1, df2])
df = df.sort_index()

print(df)