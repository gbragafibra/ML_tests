import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, GATConv, TransformerConv
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops


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

		self._initialize_weights()

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


class GAT(nn.Module):
	def __init__(self, h_dim):
		super().__init__()
		# 4 attention heads
		# once a GATConv is applied the output
		# is going to have num_heads * h_dim features' size
		self.conv1 = GATConv(4, h_dim, heads = 4)
		self.conv2 = GATConv(4 * h_dim, h_dim, heads = 4)
		self.conv3 = GATConv(4 * h_dim, h_dim, heads = 4)
		self.fc = nn.Linear(4 * h_dim, 1, bias=False)
		nn.init.xavier_uniform_(self.fc.weight)
	
	def forward(self, input_):
		x, e = input_.x, input_.edge_index
		h = self.conv1(x, e)
		h = F.relu(h)
		h = self.conv2(h, e)
		h = F.relu(h)
		h = self.conv3(h, e)
		h = F.relu(h)
		h = F.dropout(h, p = 0.5, training = self.training)
		h = global_mean_pool(h, input_.batch)
		h = self.fc(h)
		h = F.sigmoid(h)

		return h

class GTR(nn.Module):
	"""
	Using the graph transformer operator
	"""
	def __init__(self, h_dim):
		super().__init__()
		self.conv1 = TransformerConv(4, h_dim, heads = 4, beta = True)
		self.conv2 = TransformerConv(4 * h_dim, h_dim, heads = 4, beta = True)
		self.conv3 = TransformerConv(4 * h_dim, h_dim, heads = 4, beta = True)
		self.fc = nn.Linear(4 * h_dim, 1, bias = False)
		nn.init.xavier_uniform_(self.fc.weight)

	def forward(self, input_):
		x, e = input_.x, input_.edge_index
		h = self.conv1(x, e)
		h = F.relu(h)
		h = self.conv2(h, e)
		h = F.relu(h)
		h = self.conv3(h, e)
		h = F.relu(h)
		h = F.dropout(h, p = 0.5, training = self.training)
		h = global_mean_pool(h, input_.batch)
		h = self.fc(h)
		h = F.sigmoid(h)

		return h


class GINConv_(MessagePassing):
	"""
	Trying out building GINConv
	"""
	def __init__(self, in_, out_):
		super().__init__(aggr = "add")#Add AGGREGATE op
		self.MLP = nn.Sequential(
			nn.Linear(in_, out_),
			nn.ReLU(),
			nn.Linear(out_, out_))
		self.ε = nn.Parameter(torch.Tensor([0])) # can change

	def forward(self, x, edge_index, edge_attr):
		"""
		Also using edge attributes
		"""
		# Making sure Ã = A + I
		edge_index, _ = add_self_loops(edge_index, num_nodes = x.size(0))

		m_ = self.propagate(edge_index, x=x, edge_attr=edge_attr)

		#Propagate message
		m = self.MLP((1 + self.ε) * x + m_)
		return m 

	def message(self, x_j):
		return x_j

	def update(self, aggr_out):
		# With ReLU
		return F.relu(aggr_out)



class GIN_(nn.Module):
	def __init__(self, h_dim):
		super().__init__()
		self.conv1 = GINConv_(4, h_dim)
		self.conv2 = GINConv_(h_dim, h_dim)
		self.conv3 = GINConv_(h_dim, h_dim)
		self.fc = nn.Linear(h_dim, 1, bias=False)


	def forward(self, input_):
		x, e, e_attr = input_.x, input_.edge_index, input_.edge_attr

		h = self.conv1(x, e, e_attr)
		h = self.conv2(h, e, e_attr)
		h = self.conv3(h, e, e_attr)
		h = F.dropout(h, p = 0.5, training = self.training)
		h = global_mean_pool(h, input_.batch)
		h = self.fc(h)
		h = F.sigmoid(h)

		return h