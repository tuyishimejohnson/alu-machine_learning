#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('pipeline/pandas/datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
""" df2 = from_file('pipeline/pandas/datasets/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',') """

df = df.set_index("Timestamp")
print(df)