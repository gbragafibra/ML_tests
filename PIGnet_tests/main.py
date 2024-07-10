import torch
from dataset import *
from models	import *
from torch.nn.parallel import DataParallel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 


def run(model, loader, opt, loss, device, num_epochs,
	plot = False):
	
	train_loader, test_loader = train_test_split(loader, test_size = 0.2, random_state = 37)

	train_losses = []
	test_losses = []
	for e in range(num_epochs):
		#training
		model.train()
		train_loss = 0
		for i, batch in enumerate(train_loader):
			opt.zero_grad()
			batch = [sample.to(device) for sample in batch]

			output = model(batch)
			y_true = torch.cat([sample.y.unsqueeze(0) for sample in batch], dim=0).to(device)
			#y_true = y_true.view_as(output) #ensuring same dims
			l = loss(output, y_true)
			train_loss += l.item()
			print(f"Epoch [{e+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {l.item():.4f}")
			l.backward()
			opt.step()
		train_losses.append(train_loss/len(train_loader))
		print(f"Epoch [{e+1}/{num_epochs}] Average Train Loss: {train_loss / len(train_loader):.4f}")
		
		#testing
		model.eval()
		with torch.no_grad():
			test_loss = 0
			for i, batch in enumerate(test_loader):
				batch = [sample.to(device) for sample in batch]
				output = model(batch)
				y_true = torch.cat([sample.y.unsqueeze(0) for sample in batch], dim=0).to(device)
				l = loss(output, y_true)
				test_loss += l.item()
			test_losses.append(test_loss/len(test_loader))
			print(f"Epoch [{e+1}/{num_epochs}] Average Test Loss: {test_loss / len(test_loader):.4f}")

	if plot:
		plt.plot(range(num_epochs), train_losses, "ro-", label = "Train Loss")
		plt.plot(range(num_epochs), test_losses, "bo-", label = "Testing Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		plt.show()

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
		keys, data_dir, id_to_y, 12)

	in_dim = 54
	hid_dim = 200
	out_dim = 1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device = torch.device("cpu")
	model = GNN(in_dim, hid_dim, out_dim).to(device)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	opt = torch.optim.Adam(model.parameters(), lr = 0.001)
	loss = nn.MSELoss()

	run(model, list(dataloader), opt, loss, device, 10,
		plot = True)	
