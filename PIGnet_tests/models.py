import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from dataset import *

class GNN(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim):
		super().__init__()
		self.conv1 = GCNConv(in_dim, hid_dim)
		self.conv2 = GCNConv(hid_dim, hid_dim)
		self.fc = nn.Linear(hid_dim * hid_dim, out_dim)

	def forward(self, sample):
		ligand_h = sample.x_lig
		target_h = sample.x_tar
		inter_idx = sample.A_inter
		
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

		h_ = self.fc(inter)

		return h

def train(model, loader, opt, loss, device):
	model.train()
	for batch in loader:
		opt.zero_grad()
		batch = batch.to(device)

		output = model(batch)
		y_true = batch.affinity

		l = loss(output, y_true)
		print(f"Loss: {l.item()}")
		l.backward()
		opt.step()
	pass

if __name__ == "__main__":
	key_file = "coreset_keys.txt"
	keys_ = read_keys(key_file)
	#two proteins that are still NoneType
	# even when using the PDB file
	except_ = ["1gpk", "3kwa"]
	keys = [key for key in keys_ if key not in except_]
	set_file = "data/CASF-2016/power_docking/CoreSet.dat"
	data_dir = "data/MOLS/"
	id_to_y = create_key_to_y(key_file, set_file)


	dataset, dataloader = get_dataset_dataloader(
		keys, data_dir, id_to_y, 5)

	in_dim = 128
	hid_dim = 64
	out_dim = 1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = GNN(in_dim, hid_dim, out_dim).to(device)
	opt = torch.optim.Adam(model.parameters(), lr = 0.001)
	loss = nn.MSELoss()

	for epoch in range(10):
		train(model, dataloader, opt, loss, device)	