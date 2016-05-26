## in command line: $ python nntrain.py trainset testset
import sys
from time import time
from numpy import *
from nn import *

# Load data
data = loadtxt(sys.argv[1])
testset = loadtxt(sys.argv[2])
X = data[:, :2]; y = data[:, 2:]
Xtest = testset[:, :2]; ytest = testset[:, 2:]

# Setup the data matrix appropriately, and the training paramters
[m, n] = [X.shape[0], X.shape[1]]

# Add intercept term to X and X_test
X = hstack([ones((m, 1)), X])
Xtest = insert(Xtest, [0], ones((Xtest.shape[0], 1)), axis=1)
l_struc = [n, 8, 3, 1]  # d-M1-M2-1 ANN architecture, Eout avg = 0.0366
r = .1  				# weights randomly initialized in (-r,r)
eta = .01				# learning rate

# Backprop algorithm training and evaluation
t0 = time()
model = Network(l_struc, m, r)
Eout_arr = []
for exp in range(50):
	
	for iter in range(50000):
		model.forward(X)
		model.backprop(y, eta)
	# evaluation
	model.forward(Xtest)
	H = model.layers[len(l_struc)-1].output[:, 1:]
	ypred = sign(H)
	ypred[ypred == 0] = 1
	Eout = mean((ypred != ytest).astype(float))
	Eout_arr.append(Eout)
	
print "Elapsed time:", round(time() - t0, 3), "s"  # 333.653 sec
print "Averaged Eout:", mean(Eout_arr)  # 0.036