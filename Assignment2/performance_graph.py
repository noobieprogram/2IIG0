import matplotlib.pyplot as plt

x1 = ['Random / \n Euclidiean', 'Random / \n Manhattan', 'Forgy / \n Euclidean', 'Forgy / \n Manhattan', 'k-means++ / \n Euclidean', 'k-means++ / \n Manhattan']

y1 = [0.86, 0.86, 0.97, 0.73, 0.86, 0.99]

y2 = [0.795]*len(y1)

y3 = [21, 28, 5, 11, 9, 12]

y4 = [11, 11, 8, 10, 8, 11]

plt.xlabel("Configuration")
plt.ylabel("Number of iterations")
plt.plot(x1 , y3)
plt.scatter(x1, y3)

plt.plot(x1 , y4)
plt.scatter(x1, y4)

plt.show()