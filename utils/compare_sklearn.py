import random

def compare_sklearn(sklearn_alg, my_alg, lower, upper):
	m = 10
	print("The result of Sklearn and My implemtation: ")
	for i in range(m):
		test_sample = round(random.uniform(lower, upper), 4)
		print("X = %lf, predicted y are: %lf \t %lf" % (test_sample, sklearn_alg(test_sample), my_alg([test_sample])))