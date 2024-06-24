import numpy as np
from scipy.linalg import sqrtm



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
		super().__init__(self.sigmoid, self.sigmoid_prime)

	@staticmethod
	def sigmoid(x):
		return 1/(1 + np.exp(-x))

	@staticmethod
	def sigmoid_prime(x):
		σ = Sigmoid.sigmoid(x)

		return σ * (1 - σ)

class ReLU(Activation):

	def __init__(self):
		super().__init__(self.relu, self.relu_prime)

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

	def __init__(self, norma = False,
		renorma = False):
		"""
		To prevent issues of vanishing/exploding
		gradient, have better weight initialization
		"""
		self.norma = norma
		"""
		Renormalize Ã in order to prevent
		vanishing/exploding gradient problems

		https://arxiv.org/pdf/1609.02907
		"""
		self.renorma = renorma

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
		
		if self.renorma:
			self.A = A

			# https://en.wikipedia.org/wiki/Degree_matrix
			D = np.zeros(self.A.shape, dtype = int)
			np.fill_diagonal(D, np.sum(self.A, axis = 0))
			D_tilda = np.linalg.inv(sqrtm(D))

			self.A = np.dot(np.dot(D_tilda, self.A), D_tilda)
		else:
			self.A = A #dims: n_atoms x n_atoms
		

		if self.norma:
			self.W = np.random.randn(*self.A.shape)/np.sqrt(self.A.shape[0])
		else:
			self.W = np.random.randn(*self.A.shape)

		H_out = np.dot(np.dot(self.A, self.H).T, self.W)

		return H_out.T, A

	def backward(self, Δ_out, α):

		Δ_H = np.dot(Δ_out, np.dot(self.A, self.W))

		self.W -= α * np.dot(Δ_out, np.dot(self.H, self.A.T))

		return Δ_H


### Graph padding/trimming initial layer
class Resize(Layer):
	"""
	Performs a sum pooling op for trimming
	"""

	def __init__(self, λ):
		"""
		λ size to pad or trim to
		"""
		self.λ = λ

	
	def forward(self, H, A):

		N = H.shape[0]
		
		if N == self.λ:

			return H, A 

		#Trimming
		elif N > self.λ:

			pool_size = N // self.λ
			remainder = N % self.λ

			strides = [pool_size + 1 if i < remainder \
			else pool_size for i in range(self.λ)]

			H_pooled = np.array([np.sum(H[i * strides[i] : \
				(i + 1) * strides[i]]) for i in range(self.λ)])

			A_pooled = np.zeros((self.λ, self.λ))

			for i in range(self.λ):
				for j in range(self.λ):
					A_ = A[i * strides[i] : (i + 1) * strides[i],\
					j * strides[j] : (j + 1) * strides[j]]

					A_pooled[i, j] = np.sum(A_)

			#Normalize wrt min and max: H and A
			H_ = (H_pooled - np.min(H_pooled)) / (np.max(H_pooled) -\
				np.min(H_pooled))
			A_ = (A_pooled - np.min(A_pooled)) / (np.max(A_pooled) -\
				np.min(A_pooled))

			return H_, A_

		#Padding
		elif N < self.λ:
			# Padding with 0s
			pad_size = self.λ - N

			H_pad = np.pad(H, ((0, pad_size), (0,0)), \
				mode = "constant", constant_values = 0)
			
			A_pad = np.pad(A, ((0, pad_size), (0, pad_size)), \
				mode = "constant", constant_values = 0)

			return H_pad, A_pad

	def backward(self):
		"""
		Initial layer without learnable params.
		Can have null backward pass
		"""
		pass






# Mean-squared error loss func

def mse(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
	return 2 * (y_true - y_pred)/(y_true.flatten()).shape[0]

# Binary Cross-Entropy loss func

def bce(y_true, y_pred):
	# np.nan_to_num() to prevent instability issues
	return -np.mean(np.nan_to_num(y_true * np.log(y_pred) +\
		(1 - y_true) * np.log(1 - y_pred)))

def bce_prime(y_true, y_pred):
	return ((1 - y_true)/(1 - y_pred) -\
		(y_true/y_pred) / (y_true.flatten()).shape[0])	