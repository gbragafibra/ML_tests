import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
from torch.nn import functional as F


class GCN(nn.Module):
	def __init__(self, h_dim):
		super().__init__()
		# 4 node features, h_dim -> dims for hidden layers
		self.conv1 = GCNConv(4, h_dim)
		self.conv2 = GCNConv(h_dim, h_dim)
		self.conv3 = GCNConv(h_dim, h_dim)
		self.fc = nn.Linear(h_dim, 1, bias=False)
		nn.init.xavier_uniform_(self.fc.weight)

	def forward(self, input_):
		x, e = input_.x, input_.edge_index
		x = self.conv1(x, e)
		x = F.relu(x)
		x = self.conv2(x, e)
		x = F.relu(x)
		x = self.conv3(x, e)
		x = F.relu(x)
		x = F.dropout(x, p = 0.5, training = self.training)
		x = global_mean_pool(x, input_.batch)
		x = self.fc(x)
		x = F.sigmoid(x)

		return x

class GIN(nn.Module):
	def __init__(self, h_dim):
		super().__init__()
		self.conv1 = GINConv(nn.Sequential(nn.Linear(4, h_dim, bias=False),
			nn.BatchNorm1d(h_dim), nn.ReLU()))
		self.conv2 = GINConv(nn.Sequential(nn.Linear(h_dim, h_dim, bias=False),
			nn.BatchNorm1d(h_dim), nn.ReLU()))
		self.conv3 = GINConv(nn.Sequential(nn.Linear(h_dim, h_dim, bias=False),
			nn.BatchNorm1d(h_dim), nn.ReLU()))
		self.fc = nn.Linear(h_dim, 1, bias=False)

		self._initialize_weights

	def _initialize_weights():
		def init_linear(self):
			if isinstance(layer, nn.Linear):
				nn.init.xavier_uniform_(layer.weight)
		self.conv1.apply(init_linear)
		self.conv2.apply(init_linear)
		self.conv3.apply(init_linear)
		init_linear(self.fc)

	def forward(self, input_):
		x, e = input_.x, input_.edge_index

		h = self.conv1(x, e)
		h = self.conv2(h, e)
		h = self.conv3(h, e)
		h = F.dropout(h, p = 0.5, training = self.training)
		h = global_mean_pool(h, input_.batch)
		h = self.fc(h)
		h = F.sigmoid(h)

		return h