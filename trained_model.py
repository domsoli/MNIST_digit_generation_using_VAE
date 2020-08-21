import os
import sys
import torch
import scipy.io as spio
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from training_VAE import VarAutoencoder


# Testing function
def test_epoch(net, dataloader, loss_fn, device):
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()])
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


# Load data
try:
    data_path = sys.argv[1]
except:
    data_path = "./"

dataset = MNIST(data_path, download=False, transform=transforms.Compose([transforms.ToTensor(),]))

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Initialize the VarAutoencoder object
net = VarAutoencoder(encoded_space_dim=10)
# Load network parameters in evaluation mode
net.load_state_dict(torch.load('params/net_params_10.pth'))
net.to(device).eval()

test_dataloader = DataLoader(dataset, batch_size=512, shuffle=False)
# X = torch.from_numpy(X)
loss_fn = torch.nn.MSELoss()
# Compute loss
loss = test_epoch(net, test_dataloader, loss_fn, device=device)

# Print results in file
spio.savemat('MNIST.mat', {"mean_loss(MSE)": loss.tolist()})

print("MSE: ", loss.tolist())
