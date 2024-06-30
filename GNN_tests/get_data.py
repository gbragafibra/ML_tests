from rdkit import Chem
import torch
from torch.utils.data import Dataset
import pandas as pd
from torch_geometric.data import Data, DataLoader


def atom_feats(atom): # node(atom)features
	return [atom.GetAtomicNum(),
	atom.GetDegree(),
	atom.GetNumImplicitHs(),
	atom.GetIsAromatic()]

def bond_feats(bond): # edge(bond)features
	return [bond.GetBondType(),
	bond.GetStereo()]

def smi_to_PyG(smi, task_):
	"""
	Transform a mol in smiles format
	into an Tensor object for PyTorch
	geometric. Also takes a task, from
	the respective dataset to get the
	labels.
	"""
	mol = Chem.MolFromSmiles(smi)
	if mol is None:
		return None
	edges = []
	for bond in mol.GetBonds():
		i = bond.GetBeginAtomIdx()
		j = bond.GetEndAtomIdx()
		edges.extend([(i,j), (j,i)])
	edge_idx = list(zip(*edges))
	node_f = [atom_feats(a) for a in mol.GetAtoms()]
	edge_f = [bond_feats(b) for b in mol.GetBonds()]

	return Data(x=torch.Tensor(node_f),
		edge_index=torch.LongTensor(edge_idx),
		edge_attr=torch.Tensor(edge_f),
		y=torch.Tensor([task_]))

class GetData(Dataset):
	"""
	Using base class torch.Dataset
	Want to transform the respective
	dataset into Tensor compatible.
	"""
	def __init__(self, smiles, task):
		mols = [smi_to_PyG(smi, task_) for smi, task_\
		in zip(smiles, task)]
		#Get rid of invalid graphs
		self.X = [mol for mol in mols if mol]

	def __getitem__(self, idx):
		return self.X[idx]

	def __len__(self):
		return len(self.X)


def split_sets(dataset, train_size, val_size,
	test_size, batch_size):
	"""
	Takes correspondent dataset and splits
	into train, val, and test sets, with
	train, val and test sizes (∈ [0, 1]).
	Also takes batch_size (∈ N).
	"""
	N = len(dataset)
	train_loader = DataLoader(dataset[:int(N * train_size)], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(dataset[int(N * train_size):int(N * (train_size + val_size))], batch_size=batch_size, shuffle=True)
	test_loader = DataLoader(dataset[int(N * (train_size + val_size)):], batch_size=batch_size, shuffle=True)

	return train_loader, val_loader, test_loader

if __name__ == "__main__":
	sider = pd.read_csv("../data/sider.csv")
	sider_PyG = GetData(sider["smiles"], \
		sider["Hepatobiliary disorders"])
	tr_ld, val_ld, ts_ld = split_sets(sider_PyG, 0.7, 0.15, 0.15, 16)
	print(len(tr_ld))#num batches in tr_ld