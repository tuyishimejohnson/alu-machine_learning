#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
SMALL_SIZE = 8
plt.rc('font', size=SMALL_SIZE)
plt.title("All in One")



plt.subplot(3,2,1)
y = np.arange(0, 10) ** 3
x = np.arange(0, 10)

plt.plot(x, y, color="red")

# End of 0 question

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.subplot(3,2,2)
plt.scatter(x, y, color="magenta") 
plt.xlabel("Height (in)") 
plt.ylabel("Weight (lbs)") 
plt.title("Men's Height vs Weight") # The title

# End of 1 question



x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.subplot(3,2,3)
plt.xlabel("Time (years)") 
plt.ylabel("Fraction Remaining") 
plt.title("Exponential Decay of C-14") 
plt.semilogy(x,y)

# End of 2 question

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)


plt.subplot(3,2,4)
plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements") # The title
plt.plot(x,y1, label="C-14",linestyle="dashed",color="red") 
plt.plot(x,y2,label="Ra-226", color="green") 
plt.legend() # it adds the legend

# End of 3 question


np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.subplot2grid((3,2), (2,0),colspan = 2)
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.hist(student_grades,bins=10, color="#598dc9",edgecolor="black")
# End of 5 question


plt.tight_layout()
plt.show()