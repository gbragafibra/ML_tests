import numpy as np
import pandas as pd 
from rdkit import Chem

def preprocess(data, task):
	A_ = [] # Adjacency matrices
	H_ = [] # Node feature vectors
	E_ = [] # Edge feature vectors
	y_ = [] # Corresponding labels

	for idx, mol in data.iterrows():
		mol_ = Chem.MolFromSmiles(mol["smiles"])
		n_atoms = mol_.GetNumAtoms()

		A = np.zeros((n_atoms, n_atoms), dtype = int)
		E = [bond.GetBondType() for bond in mol_.GetBonds()]
		H = [atom.GetAtomicNum() for atom in mol_.GetAtoms()]

		for bond in mol_.GetBonds():
			"""
			Build Adjacency Matrix
			Self-connected -> (Ã = A + I)
			"""
			i = bond.GetBeginAtomIdx()
			j = bond.GetEndAtomIdx()

			A[i,j] = 1
			A[j,i] = 1

		np.fill_diagonal(A, 1)
		H = np.array(H).reshape(-1, 1)
		E = np.array(E).reshape(-1, 1)

		y = mol[task]
		A_.append(A)
		H_.append(H)
		E_.append(E)
		y_.append(y)

	mol_set = {
	"Adjacency Matrices" : A_,
	"Node Features" : H_,
	"Edge Features" : E_,
	"Labels" : y_
	}

	return mol_set


#Testing if it works
if __name__ == "__main__":
	dataset = pd.read_csv("data/sider.csv")
	t = "Investigations"

	mol_set = preprocess(dataset, t)

	A = mol_set["Adjacency Matrices"]
	H = mol_set["Node Features"]

	print(A[1])
	print(H[1])