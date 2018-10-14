import matplotlib.pyplot as plt

def plot_data(X, y, xlabel, ylabel):
	plt.scatter(X, y)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.show()