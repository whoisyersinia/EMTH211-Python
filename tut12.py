import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


x = np.array([0,1,2,3,4])
y = np.array([ -2,-1,2,7,14])
res = stats.linregress(x, y)

print(f"Slope: {res.slope:.4f}")
print(f"Intercept: {res.intercept:.4f}")
print(f"R-squared: {res.rvalue**2:.4f}")  # Goodness of fit

y1 = 3.5*res.slope + res.intercept
y2 = 5.5*res.slope + res.intercept
print(y1)
print(y2)

plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, res.slope * x + res.intercept, color='red', label='Least Squares Line')
plt.plot(3.5, y1, 'go')
plt.plot(5.5, y2, 'go')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

x_norm = x.dot(x)
y1 = y - (res.slope + x * res.intercept)
rss = y1.dot(y1)
se = np.sqrt((1/np.size(x)) * rss/x_norm)
confidence_interval = [res.intercept - 2.353*se,res.intercept + 2.353*se]
print(confidence_interval)
