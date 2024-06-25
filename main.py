import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from preprocess import *
from utils import *


def predict(Net, input):
	"""
	Provide a list (Net) with
	the layers, sequentially.
	And corresponding initial input.
	"""
	output = input
	for layer in Net:
		"""
		if isinstance(layer, Resize):
			output = layer.forward(*output)
		else:
		"""
		output = layer.forward(output)


	return output


# Mini-Batch Gradient Descent
def MBGD(Net, dataset, task, loss, loss_prime,
	epochs, α, batch_size, train_ω, verbose = True,
	plot = False):

	mols = preprocess(dataset, task)

	y = np.array(mols["Labels"])


	"""
	Can't make these np.arrays as
	they have arrays with inhomogeneous
	shapes.
	"""
	A_ = mols["Adjacency Matrices"]
	H_ = mols["Node Features"]


	train_size = int(train_ω * len(H_))

	idx = np.random.permutation(len(H_))
	train_idx = idx[:train_size]
	test_idx = idx[train_size:]

	train_feat = [(H_[i], A_[i]) for i in train_idx]
	train_y = y[train_idx]
	test_feat = [(H_[i], A_[i]) for i in test_idx]
	test_y = y[test_idx]
	train_losses = []
	test_losses = []

	for e in range(epochs):
		train_loss = 0
		train_correct = 0

		for i in range(0, len(train_feat), batch_size):
			x_batch = train_feat[i : i + batch_size]
			y_batch = train_y[i : i + batch_size]

			batch_loss = 0
			batch_grad = 0

			# forward pass
			for x, y in zip(x_batch, y_batch):
				output = predict(Net, x)
				batch_loss += loss(y, output)

				if (output >= 0.5 and y == 1) or\
				(output < 0.5 and y == 0):
					train_correct += 1

				batch_grad += loss_prime(y, output)

			# backward pass
			batch_grad /= len(x_batch)
			for layer in reversed(Net):
				batch_grad = layer.backward(batch_grad, α)

			train_loss += batch_loss

		test_loss = 0
		test_correct = 0
		for x, y in zip(test_feat, test_y):
			output = predict(Net, x)
			test_loss += loss(y, output)
			
			if (output >= 0.5 and y == 1) or\
				(output < 0.5 and y == 0):
					test_correct += 1

		train_losses.append(train_loss)
		test_losses.append(test_loss)
		train_acc = train_correct/len(train_feat)
		test_acc = test_correct/len(test_feat)

		if verbose:
			print(f"Epoch {e + 1}/{epochs}, \
				Training Loss = {train_loss}, \
				Training accuracy = {train_acc}")
			print(f"Epoch {e + 1}/{epochs}, \
				Testing Loss = {test_loss}, \
				Testing accuracy = {test_acc}")

	if plot:
		plt.plot(np.arange(1,epochs + 1,1), train_losses, "k", label = "Training Loss")
		plt.plot(np.arange(1,epochs + 1,1), test_losses, "r", label = "Testing Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()
	pass



if __name__ == "__main__":
	data = pd.read_csv("data/sider.csv")
	t = "Investigations"

	Net = [Resize(32),
	GIN(norma = True, renorma = True),
	ReLU(),
	GIN(norma = True, renorma = True),
	ReLU(),
	GIN(norma = True, renorma = True),
	ReLU(),
	Dense(1),
	Sigmoid()
	]


	MBGD(Net, data, t, mse, mse_prime,
	20, 0.01, 30, 0.7, verbose = True,
	plot = True)