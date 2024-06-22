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
		return np.max(0, x)

	@staticmethod
	def relu_prime(x):
		return np.where(x > 0, 1, 0)



# Mean-squared error loss func

def mse(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
	return 2 * (y_true - y_pred)/(y_true.flatten()).shape[0]

	