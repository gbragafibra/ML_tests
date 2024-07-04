import os
import pickle
from rdkit import Chem


def load_mol(path, file_type):
	"""
	Check for errors when using
	rdkit.Chem.MolFromMol2File as,
	when doing it for proteins, some
	give errors -> None type, usually
	fixed with rdkit.Chem.MolFromPdbFile
	"""
	try:
		if file_type == "mol2":
			return Chem.MolFromMol2File(path)
		elif file_type == "pdb":
			return Chem.MolFromPDBFile(path)
		else:
			raise ValueError(f"Unsupported file type: {file_type}")
	except Exception as e:
		print(f"Error loading molecule from {path}: {e}")
		return None

def serialize_mols(file, data_dir, output_dir):
	# read keys from txt file
	with open(file, 'r') as f:
		keys = [line.strip() for line in f.readlines() if line.strip()]

	for key in keys:
		subdir_path = os.path.join(data_dir, key)

		if not os.path.exists(subdir_path):
			print(f"Warning: Directory {subdir_path} does not exist for key {key}. Skipping.")
			continue

		ligand_file = os.path.join(subdir_path, f"{key}_ligand.mol2")
		protein_file = os.path.join(subdir_path, f"{key}_protein.mol2")

		if not os.path.isfile(ligand_file) or not os.path.isfile(protein_file):
			print(f"Warning: Missing mol2 files for key {key}. Skipping.")
			continue

		ligand_mol = load_mol(ligand_file, "mol2")
		protein_mol = load_mol(protein_file, "mol2")

		if protein_mol is None:
			print(f"Loading from PDB for key {key} due to NoneType in MOL2 files, for the target.")
			protein_pdb_file = os.path.join(subdir_path, f"{key}_protein.pdb")
			protein_mol = load_mol(protein_pdb_file, "pdb")



		# serialize both lig and target into one file
		output_file = os.path.join(output_dir, f"{key}_molecules.pkl")
		with open(output_file, 'wb') as f:
			pickle.dump((ligand_mol, protein_mol), f)

		print(f"Serialized molecules for key {key} to {output_file}")




if __name__ == "__main__":
	txt_path = "coreset_keys.txt"
	data_dir = "data/CASF-2016/coreset/"
	output = "data/MOLS/"
	serialize_mols(txt_path, data_dir, output)