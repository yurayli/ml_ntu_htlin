import numpy as np

class Layer:
	
	def __init__(self, id, m, n_neurons, prev_neurons):
		self.id = id
		self.n_neurons = n_neurons
		self.prev_neurons = prev_neurons
		# initialization
		self.input = np.zeros((m, n_neurons))
		self.output = np.zeros((m, n_neurons))
		self.error = np.zeros((m, n_neurons))
		self.weight = np.zeros((prev_neurons+1, n_neurons))
		
	def setWeight(self, r):
		self.weight = -r + 2 * r * np.random.rand(self.prev_neurons+1, self.n_neurons)
		
		
class Network:
## Input arguments:
## 		l_neurons = list of neurons of each layer, 
## 		m = number of examples, r = bound of initialized weights
## 		X = features, y = labels, eta = learning rate
## self.layers = list of each layer within the NN

	def __init__(self, l_neurons, m, r):
		self.layers = []
		for l in range(len(l_neurons)):
			n_neurons = l_neurons[l]
			prev_neurons = 0 if l == 0 else l_neurons[l-1]
			layer = Layer(l, m, n_neurons, prev_neurons)
			layer.setWeight(r)
			self.layers.append(layer)
			
	def forward(self, X):
		self.layers[0].output = X
		m = X.shape[0]
		for l in range(1,len(self.layers)):
			self.layers[l].input = np.dot(self.layers[l-1].output, self.layers[l].weight)
			self.layers[l].output = np.tanh(self.layers[l].input)
			self.layers[l].output = np.hstack([np.ones((m,1)), self.layers[l].output])
	
	def backprop(self, y, eta):
		L = len(self.layers)-1
		H = self.layers[L].output[:, 1:]
		self.layers[L].error = -2 * (y - H) * (1 - H ** 2)
		grad = np.dot(self.layers[L-1].output.T, self.layers[L].error)
		self.layers[L].weight -= eta * grad
		
		for l in range(L-1, 0, -1):
			if l == (L-1):
				self.layers[l].error = np.dot(self.layers[l+1].error, self.layers[l+1].weight.T) * (1 - self.layers[l].output ** 2)
			else:
				self.layers[l].error = np.dot(self.layers[l+1].error[:, 1:], self.layers[l+1].weight.T) * (1 - self.layers[l].output ** 2)
			grad = np.dot(self.layers[l-1].output.T, self.layers[l].error[:, 1:])
			self.layers[l].weight -= eta * grad;
		
