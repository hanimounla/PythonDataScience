import matplotlib.pyplot as plt #this library is the most used for data plotting and usualy call it plt


plt.plot([1,2,3],[4,5,3], color = 'b', marker = 'o', label = 'line1') #first line (1,4),(2,5),(3,3)
plt.plot([1,2,3],[5,3,4], color = 'r', marker = '*', label = 'line2') #second line (1,5),(2,3),(3,4)

plt.legend() #to display the legend for the chart
plt.show() # show the chart