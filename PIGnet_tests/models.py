import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_mean_pool


class GNN(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim):
		super().__init__()
		self.conv1 = GCNConv(in_dim, hid_dim)
		self.conv2 = GCNConv(hid_dim, hid_dim)
		self.conv_inter1 = GCNConv(1, hid_dim)
		self.conv_inter2 = GCNConv(1, hid_dim)
		self.conv_inter3 = GCNConv(1, hid_dim)
		self.fc = nn.Linear(hid_dim, out_dim)


	def forward(self, batch):
		out_ = []
		for sample in batch:
			ligand_h = sample.x_lig
			target_h = sample.x_tar
			inter_idx1 = sample.A_inter[0]
			inter_idx2 = sample.A_inter[1]
			inter_idx3 = sample.A_inter[2]

			ligand_e_idx = sample.lig_e_idx
			target_e_idx = sample.tar_e_idx

			# For ligand graph
			h1 = F.relu(self.conv1(ligand_h, ligand_e_idx))
			h1 = F.relu(self.conv2(h1, ligand_e_idx))
			h1 = global_mean_pool(h1, torch.zeros(ligand_h.size(0), dtype = torch.long))

			# For protein graph
			h2 = F.relu(self.conv1(target_h, target_e_idx))
			h2 = F.relu(self.conv2(h2, target_e_idx))
			h2 = global_mean_pool(h2, torch.zeros(target_h.size(0), dtype = torch.long))


			# Compute interactions between lig and tar node features (outer prod)
			inter = torch.einsum("bi,bj->bij", h1, h2)
			#flatten
			inter = inter.view(inter.size(0), -1)
			inter1 = F.relu(self.conv_inter1(inter.T, inter_idx1))
			inter2 = F.relu(self.conv_inter2(inter.T, inter_idx2))
			inter3 = F.relu(self.conv_inter3(inter.T, inter_idx3))

			inter_ = torch.stack([inter1, inter2, inter3], dim=1)
			inter_ = torch.mean(inter_, dim=1)
			inter_ = torch.mean(inter_.transpose(0, 1), dim = 1, keepdim = True)
			h_ = self.fc(inter_.T)
			out_.append(h_)

		return torch.cat(out_)

