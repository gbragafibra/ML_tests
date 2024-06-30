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
		self.conv4 = GCNConv(h_dim, h_dim)
		self.fc = nn.Linear(h_dim, 1)

	def forward(self, input_):
		x, e = input_.x, input_.edge_index
		x = self.conv1(x, e)
		x = F.relu(x)
		x = self.conv2(x, e)
		x = F.relu(x)
		x = self.conv3(x, e)
		x = F.relu(x)
		x = self.conv4(x, e)
		x = F.relu(x)
		x = global_mean_pool(x, input_.batch)
		x = self.fc(x)
		x = F.sigmoid(x)

		return x