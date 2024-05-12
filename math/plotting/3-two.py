#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

plt.xlabel("Time (years)") # This line shows time in yeaars on x-axis
plt.ylabel("Fraction Remaining") # This line shows fraction remaining on y-axis
plt.title("Exponential Decay of Radioactive Elements") # The title
plt.plot(x,y1, label="C-14",linestyle="dashed",color="red") #Creating the plot for C-14
plt.plot(x,y2,label="Ra-226", color="green") #Creating the plot for Ra-226
plt.legend() # it adds the legend
plt.show() # displays the plot