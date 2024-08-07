import pickle
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import Atom, Mol
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds
import numpy as np
import torch
import random
#from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch, DataLoader
from typing import List, Dict, Tuple, Any
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
random.seed(37)
RDLogger.DisableLog("rdApp.*")

INTERACTION_TYPES = [
	"hbonds",
	"hydrophobic",
	"metal_complexes",
]

#Periodic table
pt = """
H,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,HE
LI,BE,1,1,1,1,1,1,1,1,1,1,B,C,N,O,F,NE
NA,MG,1,1,1,1,1,1,1,1,1,1,AL,SI,P,S,CL,AR
K,CA,SC,TI,V,CR,MN,FE,CO,NI,CU,ZN,GA,GE,AS,SE,BR,KR
RB,SR,Y,ZR,NB,MO,TC,RU,RH,PD,AG,CD,IN,SN,SB,TE,I,XE
CS,BA,LU,HF,TA,W,RE,OS,IR,PT,AU,HG,TL,PB,BI,PO,AT,RN
"""
PERIODIC_TABLE = dict()
for i, per in enumerate(pt.split()):
	for j, ele in enumerate(per.split(",")):
		PERIODIC_TABLE[ele] = (i, j)

PERIODS = [0, 1, 2, 3, 4, 5]
GROUPS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
SYMBOLS = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "X"]
DEGREES = [0, 1, 2, 3, 4, 5]
HYBRIDIZATIONS = [
	Chem.rdchem.HybridizationType.S,
	Chem.rdchem.HybridizationType.SP,
	Chem.rdchem.HybridizationType.SP2,
	Chem.rdchem.HybridizationType.SP3,
	Chem.rdchem.HybridizationType.SP3D,
	Chem.rdchem.HybridizationType.SP3D2,
	Chem.rdchem.HybridizationType.UNSPECIFIED,
]
FORMALCHARGES = [-2, -1, 0, 1, 2, 3, 4]
METALS = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
HYDROPHOBICS = ["F", "CL", "BR", "I"]

HBOND_DONOR_INDICES = ["[!#6;!H0]"]
HBOND_ACCEPPTOR_SMARTS = [
	"[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"
]

def one_of_k_encoding(x: Any, allowable_set: List[Any]) -> List[bool]:
	if x not in allowable_set:
		raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
	return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x: Any, allowable_set: List[Any]) -> List[bool]:
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))

def get_period_group(atom: Atom) -> List[bool]:
	period, group = PERIODIC_TABLE[atom.GetSymbol().upper()]
	return one_of_k_encoding(period, PERIODS) + one_of_k_encoding(group, GROUPS)

def atom_feature(mol: Mol, atom_index: int) -> np.ndarray:
	atom = mol.GetAtomWithIdx(atom_index)
	return np.array(
		one_of_k_encoding_unk(atom.GetSymbol(), SYMBOLS)
		+ one_of_k_encoding_unk(atom.GetDegree(), DEGREES)
		+ one_of_k_encoding_unk(atom.GetHybridization(), HYBRIDIZATIONS)
		+ one_of_k_encoding_unk(atom.GetFormalCharge(), FORMALCHARGES)
		+ get_period_group(atom)
		+ [atom.GetIsAromatic()]
	)  # (9, 6, 7, 7, 24, 1) --> total 54

def get_atom_feature(mol: Mol) -> np.ndarray:
	natoms = mol.GetNumAtoms()
	H = []
	for idx in range(natoms):
		H.append(atom_feature(mol, idx))
	H = np.array(H)
	return H  


def get_hydrophobic_atom(mol: Mol) -> np.ndarray:
	natoms = mol.GetNumAtoms()
	hydrophobic_indice = np.zeros((natoms,))
	for atom_idx in range(natoms):
		atom = mol.GetAtomWithIdx(atom_idx)
		symbol = atom.GetSymbol()
		if symbol.upper() in HYDROPHOBICS:
			hydrophobic_indice[atom_idx] = 1
		elif symbol.upper() in ["C"]:
			neighbors = [x.GetSymbol() for x in atom.GetNeighbors()]
			neighbors_wo_c = list(set(neighbors) - set(["C"]))
			if len(neighbors_wo_c) == 0:
				hydrophobic_indice[atom_idx] = 1
	return hydrophobic_indice

def get_A_hydrophobic(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
	ligand_indice = get_hydrophobic_atom(ligand_mol)
	target_indice = get_hydrophobic_atom(target_mol)
	return np.outer(ligand_indice, target_indice)


def get_hbond_atom_indices(mol: Mol, smarts_list: List[str]) -> np.ndarray:
	indice = []
	for smarts in smarts_list:
		smarts = Chem.MolFromSmarts(smarts)
		indice += [idx[0] for idx in mol.GetSubstructMatches(smarts)]
	indice = np.array(indice)
	return indice

def get_A_hbond(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
	ligand_h_acc_indice = get_hbond_atom_indices(ligand_mol, HBOND_ACCEPPTOR_SMARTS)
	target_h_acc_indice = get_hbond_atom_indices(target_mol, HBOND_ACCEPPTOR_SMARTS)
	ligand_h_donor_indice = get_hbond_atom_indices(ligand_mol, HBOND_DONOR_INDICES)
	target_h_donor_indice = get_hbond_atom_indices(target_mol, HBOND_DONOR_INDICES)

	hbond_indice = np.zeros((ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms()))
	for i in ligand_h_acc_indice:
		for j in target_h_donor_indice:
			hbond_indice[i, j] = 1
	for i in ligand_h_donor_indice:
		for j in target_h_acc_indice:
			hbond_indice[i, j] = 1
	return hbond_indice

def get_A_metal_complexes(ligand_mol: Mol, target_mol: Mol) -> np.ndarray:
	ligand_h_acc_indice = get_hbond_atom_indices(ligand_mol, HBOND_ACCEPPTOR_SMARTS)
	target_h_acc_indice = get_hbond_atom_indices(target_mol, HBOND_ACCEPPTOR_SMARTS)
	ligand_metal_indice = np.array(
		[
			idx
			for idx in range(ligand_mol.GetNumAtoms())
			if ligand_mol.GetAtomWithIdx(i).GetSymbol() in METALS
		]
	)
	target_metal_indice = np.array(
		[
			idx
			for idx in range(target_mol.GetNumAtoms())
			if target_mol.GetAtomWithIdx(i).GetSymbol() in METALS
		]
	)

	metal_indice = np.zeros((ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms()))
	for ligand_idx in ligand_h_acc_indice:
		for target_idx in target_metal_indice:
			metal_indice[ligand_idx, target_idx] = 1
	for ligand_idx in ligand_metal_indice:
		for target_idx in target_h_acc_indice:
			metal_indice[ligand_idx, target_idx] = 1
	return metal_indice

def mol_to_feature(ligand_mol: Mol, target_mol: Mol) -> Data:
	# Remove hydrogens
	ligand_mol = Chem.RemoveHs(ligand_mol)
	target_mol = Chem.RemoveHs(target_mol)

	# prepare ligand
	ligand_natoms = ligand_mol.GetNumAtoms()
	ligand_adj = GetAdjacencyMatrix(ligand_mol) + np.eye(ligand_natoms)
	ligand_h = get_atom_feature(ligand_mol)

	# prepare protein
	target_natoms = target_mol.GetNumAtoms()
	target_adj = GetAdjacencyMatrix(target_mol) + np.eye(target_natoms)
	target_h = get_atom_feature(target_mol)

	interaction_indice = np.zeros(
		(len(INTERACTION_TYPES), ligand_mol.GetNumAtoms(), target_mol.GetNumAtoms())
	)
	interaction_indice[0] = get_A_hbond(ligand_mol, target_mol)
	interaction_indice[1] = get_A_metal_complexes(ligand_mol, target_mol)
	interaction_indice[2] = get_A_hydrophobic(ligand_mol, target_mol)

	ligand_e_idx = torch.tensor(ligand_adj, dtype=torch.long).nonzero(as_tuple=False).t().contiguous()
	target_e_idx = torch.tensor(target_adj, dtype=torch.long).nonzero(as_tuple=False).t().contiguous()

	A_inter = []
	for i in range(len(INTERACTION_TYPES)):
		A_inter.append(torch.tensor(interaction_indice[i], dtype=torch.long).nonzero(as_tuple=False).t().contiguous())

	sample = Data(
		x_lig = torch.tensor(ligand_h, dtype=torch.float),
		x_tar = torch.tensor(target_h, dtype=torch.float),
		A_inter = A_inter,
		lig_e_idx = ligand_e_idx,
		tar_e_idx = target_e_idx,
		)

	return sample



def read_keys(path: str) -> List[str]:
	with open(path, "r") as f:
		keys = f.read().splitlines()

	return keys


# To parse CoreSet.data in order to get affinities
def parse_set(path: str) -> Dict[str, float]:
	id_to_y = {}
	with open(path, "r") as f:
		next(f) # skip the header
		for line in f: 
			parts = line.strip().split()
			key = parts[0]
			logKa = float(parts[3])
			id_to_y[key] = logKa
	return id_to_y

def create_key_to_y(key_file: str, set_file: str) -> Dict[str, float]:
	keys = read_keys(key_file)
	all_id_to_y = parse_set(set_file)
	id_to_y = {key: all_id_to_y[key] for key in keys if key in all_id_to_y}
	return id_to_y



class ComplexDataset(Dataset):
	def __init__(self, keys: List[str], data_dir: str, id_to_y: Dict[str, float]):
		self.keys = keys 
		self.data_dir = data_dir
		self.id_to_y = id_to_y
		self.samples = []

		for key in self.keys:
			with open(os.path.join(self.data_dir + "/" + key + "_molecules.pkl"), "rb") as f:
				m1, m2 = pickle.load(f)

			sample = mol_to_feature(m1, m2)
			sample.y = torch.tensor([self.id_to_y[key] * -1.36], dtype=torch.float)
			self.samples.append(sample)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Any:
		return self.samples[idx]


def get_dataset_dataloader(keys: List[str],
	data_dir: str,
	id_to_y: Dict[str, float],
	batch_size: int):

	dataset = ComplexDataset(keys, data_dir, id_to_y)
	dataloader = loader(dataset,
		batch_size,
		shuffle = True)
	return dataloader

#Alternative loader ; without padding or trimming
def loader(dataset, batch_size, shuffle = True):
	dataset = list(dataset)
	if shuffle:
		random.shuffle(dataset)

	N = len(dataset)
	n_batches = (N + batch_size - 1) // batch_size

	for i in range(n_batches):
		start_idx = i * batch_size
		end_idx = min((i + 1) * batch_size, N)
		batch_ = dataset[start_idx:end_idx]


		yield batch_


# Alternative function to standard collate_fn
# in DataLoader (Not working properly)
def collate(batch):

	#Get max lenghts for all attributes in Data Object
	x_lig_len = max(item.x_lig.shape[0] for item in batch)
	x_tar_len = max(item.x_tar.shape[0] for item in batch)
	lig_e_idx_len = max(item.lig_e_idx.shape[1] for item in batch)
	tar_e_idx_len = max(item.tar_e_idx.shape[1] for item in batch)

	batch_x_lig = []
	batch_x_tar = []
	batch_lig_e_idx = []
	batch_tar_e_idx = []
	batch_A_inter = [[] for _ in range(3)] # 3 is len(INTERACTION_TYPES)
	batch_y = []

	for item in batch:
		batch_y.append(item.y)

		x_lig_pad_len = x_lig_len - item.x_lig.shape[0]
		batch_x_lig.append(torch.cat([item.x_lig, torch.zeros((x_lig_pad_len, item.x_lig.shape[1]))], dim=0))

		x_tar_pad_len = x_tar_len - item.x_tar.shape[0]
		batch_x_tar.append(torch.cat([item.x_tar, torch.zeros((x_tar_pad_len, item.x_tar.shape[1]))], dim=0))

		lig_e_idx_pad_len = lig_e_idx_len - item.lig_e_idx.shape[1]
		batch_lig_e_idx.append(torch.cat([item.lig_e_idx, torch.zeros((2, lig_e_idx_pad_len), dtype=torch.long)], dim=1))

		tar_e_idx_pad_len = tar_e_idx_len - item.tar_e_idx.shape[1]
		batch_tar_e_idx.append(torch.cat([item.tar_e_idx, torch.zeros((2, tar_e_idx_pad_len), dtype=torch.long)], dim=1))
		
		for i in range(3):  # 3 is len(INTERACTION_TYPES)
			A_inter_i = item.A_inter[i]
			A_inter_pad = torch.cat([
				torch.cat([A_inter_i, torch.zeros((A_inter_i.shape[0], x_tar_pad_len), dtype=A_inter_i.dtype)], dim=1),
				torch.zeros((x_lig_pad_len, A_inter_i.shape[1] + x_tar_pad_len), dtype=A_inter_i.dtype)
			], dim=0)
			batch_A_inter[i].append(A_inter_pad)

	return Data(x_lig = torch.stack(batch_x_lig),
		x_tar = torch.stack(batch_x_tar),
		A_inter = [torch.stack(a_inter) for a_inter in batch_A_inter],
		lig_e_idx = torch.stack(batch_lig_e_idx),
		tar_e_idx = torch.stack(batch_tar_e_idx),
		y = torch.tensor(batch_y))


if __name__ == "__main__":
	key_file = "coreset_keys.txt"
	keys_ = read_keys(key_file)[:20]
	#two proteins that are still NoneType
	# even when using the PDB file
	except_ = ["1gpk", "3kwa"]
	keys = [key for key in keys_ if key not in except_]
	set_file = "data/CASF-2016/power_docking/CoreSet.dat"
	data_dir = "data/MOLS/"
	id_to_y = create_key_to_y(key_file, set_file)
	

	dataloader = get_dataset_dataloader(
		keys, data_dir, id_to_y, 10)
	for i, batch in enumerate(dataloader):
		print(i, batch)
		if i >= 1:
			break