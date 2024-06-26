#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.hist(student_grades,bins=10, color="#598dc9",edgecolor="black")

plt.show()
