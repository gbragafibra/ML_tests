import numpy as np


# General Layer class

class Layer:

	def __init__(self):
		self.input = None
		self.output = None


	def forward(self, input):
		pass

	def backward(self, Δ_out, α):
		"""
		Δ_out is err in output layer
		α -> learning rate
		"""
		pass


#Activation layer base class

class Activation(Layer):

	def __init__(self, act, act_prime):
		self.act = act #Activation
		self.act_prime = act_prime #Derivative of act

	def forward(self, input):
		self.input = input

		return self.act(self.input)

	def backward(self, Δ_out, α):
		return Δ_out * self.act_prime(self.input)


class Sigmoid(Activation):

	def __init__(self):
		super().__init__(sigmoid, sigmoid_prime)

	@staticmethod
	def sigmoid(x):
		return 1/(1 + np.exp(-x))

	@staticmethod
	def sigmoid_prime(x):
		σ = Sigmoid.sigmoid(x)

		return σ * (1 - σ)

class ReLU(Activation):

	def __init__(self):
		super().__init__(relu, relu_prime)

	@staticmethod
	def relu(x):
		return np.maximum(0, x)

	@staticmethod
	def relu_prime(x):
		return np.where(x > 0, 1, 0)



### GIN layer

class GIN(Layer):
	"""
	For now doesn't take into account
	edge embeddings, nor their update.
	Might want to fuse them into the
	Adjacency matrix.
	"""

	def __init__(self, norma = False):
		"""
		To prevent issues of vanishing/exploding
		gradient, have better weight initialization
		"""
		self.norma = norma
		

	def forward(self, H, A):
		"""
		H is node embedding vectors
		for all of the graph.
		A is the self-connected adjacency
		matrix (Ã = A + I)
		"""
		"""
		Sum AGGREGATION op is implicit
		in the dot product A @ H.
		Keep in mind that MEAN AGG op is
		still possible, but MAX and MIN
		are much more difficult if not
		impossible to do in a vectorized
		manner.
		"""
		self.H = H #dims: n_atoms x 1
		self.A = A #dims: n_atoms x n_atoms
		if self.norma:
			self.W = np.random.randn(*A.shape)/np.sqrt(A.shape[0])
		else:
			self.W = np.random.randn(*A.shape)

		H_out = np.dot(np.dot(self.A, self.H).T, self.W)

		return H_out.T

	def backward(self, Δ_out, α):

		Δ_H = np.dot(Δ_out, np.dot(self.A, self.W))

		self.W -= α * np.dot(Δ_out, np.dot(self.H, self.A.T))

		return Δ_H

# Mean-squared error loss func

def mse(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
	return 2 * (y_true - y_pred)/(y_true.flatten()).shape[0]

