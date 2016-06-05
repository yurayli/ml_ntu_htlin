## in command line: $ python nntrain.py trainset testset
import sys
from time import time
import numpy as np

# Load data
data = np.loadtxt(sys.argv[1])
testset = np.loadtxt(sys.argv[2])
X = data[:, :2]; y = data[:, 2:]
Xtest = testset[:, :2]; ytest = testset[:, 2:]

# Setup the data matrix appropriately, and the training paramters
[m, n] = [X.shape[0], X.shape[1]]

# Add intercept term to X and X_test
X = np.hstack([np.ones((m, 1)), X])
Xtest = np.insert(Xtest, [0], np.ones((Xtest.shape[0], 1)), axis=1)
M1 = 8					# d-M1-M2-1 ANN architecture, Eout avg = 0.0366
M2 = 3
r = .1  				# weights randomly initialized in (-r,r)
eta = .01				# learning rate

# Backprop algorithm training and evaluation
Eout_arr = []
t0 = time()
for exp in range(50):
	w1 = -r + 2*r*np.random.rand(n+1, M1)
	w2 = -r + 2*r*np.random.rand(M1+1, M2)
	w3 = -r + 2*r*np.random.rand(M2+1, 1)
	for iter in range(50000):
		#
		hidden1 = np.tanh(np.dot(X, w1))
		hidden1 = np.hstack([np.ones((m,1)), hidden1])
		hidden2 = np.tanh(np.dot(hidden1, w2))
		hidden2 = np.hstack([np.ones((m,1)), hidden2])
		H = np.tanh(np.dot(hidden2, w3))
		#
		delH = -2 * (y-H) * (1-H**2)
		delHidden2 = np.dot(delH, w3.T) * (1 - hidden2**2)
		delHidden1 = np.dot(delHidden2[:, 1:], w2.T) * (1 - hidden1**2)
		grad3 = np.dot(hidden2.T, delH)
		grad2 = np.dot(hidden1.T, delHidden2[:, 1:])
		grad1 = np.dot(X.T, delHidden1[:, 1:])
		#
		w1 -= eta * grad1
		w2 -= eta * grad2
		w3 -= eta * grad3
	
	# evaluation
	hidden1 = np.tanh(np.dot(Xtest, w1))
	hidden1 = np.hstack([np.ones((Xtest.shape[0],1)), hidden1])
	hidden2 = np.tanh(np.dot(hidden1, w2))
	hidden2 = np.hstack([np.ones((Xtest.shape[0],1)), hidden2])
	H = np.tanh(np.dot(hidden2, w3))
	ypred = np.sign(H)
	ypred[ypred == 0] = 1
	Eout = np.mean((ypred != ytest).astype(float))
	Eout_arr.append(Eout)
	
print "Elapsed time:", round(time() - t0, 3), "s"  # 263.729 sec
print "Averaged Eout:", np.mean(Eout_arr)  # 0.03664