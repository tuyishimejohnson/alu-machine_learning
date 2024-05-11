#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180


plt.scatter(x, y, color="#16a34a")  # This line creates the scatter plot
plt.xlabel("weight") # This line shows Men's weight on x-axis
plt.ylabel("height") # This line shows Men's Height on y-axis
plt.title("Men's Height vs Weight") # The title
plt.show()