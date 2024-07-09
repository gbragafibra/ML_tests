import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from dataset import *
from torch.nn.parallel import DataParallel


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

def train(model, loader, opt, loss, device, num_epochs):
	model.train()
	for e in range(num_epochs):
		e_loss = 0
		for i, batch in enumerate(loader):
			opt.zero_grad()
			batch = [sample.to(device) for sample in batch]

			output = model(batch)
			y_true = torch.cat([sample.y.unsqueeze(0) for sample in batch], dim=0).to(device)
			#y_true = y_true.view_as(output) #ensuring same dims
			l = loss(output, y_true)
			e_loss += l.item()
			print(f"Epoch [{e+1}/{num_epochs}], Batch [{i+1}/{len(loader)}], Loss: {l.item():.4f}")
			l.backward()
			opt.step()
		print(f"Epoch [{e+1}/{num_epochs}] Average Loss: {e_loss / len(loader):.4f}")
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


	dataloader = get_dataset_dataloader(
		keys, data_dir, id_to_y, 5)

	in_dim = 54
	hid_dim = 200
	out_dim = 1

	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	device = torch.device("cpu")
	model = GNN(in_dim, hid_dim, out_dim).to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	opt = torch.optim.Adam(model.parameters(), lr = 0.001)
	loss = nn.MSELoss()

	train(model, list(dataloader), opt, loss, device, 10)	
