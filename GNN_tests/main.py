from get_data import *
from models import *
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import pandas as pd 
from torch.optim import SGD



def train_val_test(dataset, model, epochs, lr, h_dim, plot = False):
	mod = model(h_dim)
	tr_loader, val_loader, ts_loader = split_sets(dataset, 0.7, 0.15, 0.15, 16)
	opt = SGD(mod.parameters(), lr = lr)
	l = nn.MSELoss() #CrossEntropyLoss is weird, gives with GCN module always loss = 0
	train_losses = []
	val_losses = []
	test_losses = []
	
	for e in range(epochs):
		print(f"Epoch {e + 1} / {epochs}")
		#training
		mod.train()
		train_total = len(tr_loader.dataset)
		train_corr = 0
		for _, batch in enumerate(tr_loader):
			opt.zero_grad()
			pred = mod(batch)
			loss = l(pred, batch.y.reshape(-1, 1))
			loss.backward()
			opt.step()
			train_losses.append(loss.item())
			train_corr += ((pred > 0.5).float() == batch.y.reshape(-1, 1)).sum()
		train_acc = train_corr / train_total
		print(f"Training accuracy: {train_acc:.4f}")

		#validation
		mod.eval()
		val_total = len(val_loader.dataset)
		val_corr = 0
		with torch.no_grad():
			val_loss = 0 
			for batch in val_loader:
				pred = mod(batch)
				loss = l(pred, batch.y.reshape(-1, 1))
				val_loss += loss.item()
				val_corr += ((pred > 0.5).float() == batch.y.reshape(-1, 1)).sum()
			val_loss /= len(val_loader)
			val_losses.append(val_loss)
			val_acc = val_corr / val_total
			print(f"Validation accuracy: {val_acc:.4f}")

		#testing
		test_loss = 0
		test_total = len(ts_loader.dataset)
		test_corr = 0
		with torch.no_grad():
			for batch in ts_loader:
				pred = mod(batch)
				loss = l(pred, batch.y.reshape(-1, 1))
				test_loss += loss.item()
				test_corr += ((pred > 0.5).float() == batch.y.reshape(-1, 1)).sum()
			test_loss /= len(ts_loader)
			test_losses.append(test_loss)
			test_acc = test_corr / test_total
			print(f"Testing accuracy: {test_acc:.4f}")

	#Avg for each epoch
	train_losses_avg = np.array(train_losses).reshape(epochs, -1).mean(axis = 1)

	if plot:
		plt.plot(range(epochs), train_losses_avg, "ko--", label = "Training loss")
		plt.plot(range(epochs), val_losses, "bo--", label = "Validation loss")
		plt.plot(range(epochs), test_losses, "ro--", label = "Testing loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()

	pass


if __name__ == "__main__":
	sider = pd.read_csv("../data/sider.csv")
	sider_PyG = GetData(sider["smiles"], \
		sider["Eye disorders"])
	train_val_test(sider_PyG, GCN, 100, 0.05, 16, plot = True)