#!/usr/bin/env python3

# Import numpy
import numpy as np

# import '2-from_file'
from_file = __import__('2-from_file').from_file

df = from_file('pipeline/pandas/datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
last_rows = df[["High", "Close"]].tail(10)
A = np.array(last_rows)
print(A)
