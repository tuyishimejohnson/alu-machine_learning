""" #!/usr/bin/env python3 """

"""  Sort dataframe by 'High' column in descending order """

from_file = __import__('2-from_file').from_file

df = from_file('pipeline/pandas/datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df = df.sort_values(by='High', ascending=False)

print(df.head())
