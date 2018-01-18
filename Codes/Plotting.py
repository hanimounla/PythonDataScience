import matplotlib.pyplot as plt

x_range = range(10)
y_range = range(10)
plt.yticks(x_range)
plt.xticks(y_range)
plt.plot([1,2,3],[4,6,5], color = 'b', marker = 'o', label = 'line1')
plt.plot([1,2,3],[5,3,2], color = 'r', marker = '*', label = 'line2')
plt.legend()
plt.show()