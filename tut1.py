import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 11)  #11 points between start and stop ,
y = np.sin(x) + x ** 3 / 2 - x
fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
