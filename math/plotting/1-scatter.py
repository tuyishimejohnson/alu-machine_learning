#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180


plt.scatter(x, y, color="magenta")  # This line creates the scatter plot
plt.xlabel("Height (in)") # This line shows Men's height on x-axis
plt.ylabel("Weight (lbs)") # This line shows Men's weight on y-axis
plt.title("Men's Height vs Weight") # The title
plt.show()
