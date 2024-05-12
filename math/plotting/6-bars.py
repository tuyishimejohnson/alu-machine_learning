#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))
plt.ylabel("Quantity of Fruits")
plt.yticks(np.arange(0, 81, 10))
plt.title("Number of Fruit per Person")

people = ['Farrah', 'Fred', 'Felicia']
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
fruit_names = ['apples', 'bananas', 'oranges', 'peaches']()


plt.bar(people, fruit[0], color=colors[0], width=0.5, label=fruit_names[0])
plt.bar(people, fruit[0])
plt.bar(people, fruit[1], bottom=fruit[0], color=colors[1], width=0.5, label=fruit_names[1])
plt.bar(people, fruit[2], bottom=np.sum(fruit[:2], axis=0), color=colors[2], width=0.5, label=fruit_names[2])
plt.bar(people, fruit[3], bottom=np.sum(fruit[:3], axis=0), color=colors[3], width=0.5, label=fruit_names[3])

plt.legend()
plt.show()