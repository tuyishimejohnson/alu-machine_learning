#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)


plt.xlabel("Time (years)") # This line shows time in yeaars on x-axis
plt.ylabel("Fraction Remaining") # This line shows fraction remaining on y-axis
plt.title("Exponential Decay of C-14") # The title
plt.semilogy(x,y) # Plots  x and y with a straight line
plt.show()