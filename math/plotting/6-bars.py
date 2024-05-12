#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
plt.ylabel('Quantity of Fruit')
plt.yticks(np.arange(0, 81, 10))
plt.title('Number of Fruit per Person')