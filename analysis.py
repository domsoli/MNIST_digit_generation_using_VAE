import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dataset import add_args_to_string
from training_VAE import VarAutoencoder
from dataset import DoubleMnistDataset, AddGaussianNoise, AddOcclusion

# Constants
ENCODED_SPACE_DIM=6

# Define paths
data_root_dir = '../datasets'
out_path = 'datasets/'
plots_dir = 'plots/'
params_dir = 'params/'

# Parse arguments
parser = argparse.ArgumentParser(description='Analyze the latent space.')
parser.add_argument('--add_noise', type=float, default=0, help='Variance of Gaussian noise added over the input image' )
parser.add_argument('--add_occlusion', type=int, default=0, help='Number of occluded rows over the input image')
parser.add_argument('--encoded_dim', type=int, default=ENCODED_SPACE_DIM, help='Encoded space dimension')
# parse input arguments
args = parser.parse_args()

# Initialize the VarAutoencoder object
net = VarAutoencoder(encoded_space_dim=args.encoded_dim)

# Select device
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
print('Using device:', device)
# Load network parameters
net.load_state_dict(torch.load(params_dir+'net_params_{}.pth'.format(args.encoded_dim), map_location=device_name))
# Move all the network parameters to the selected device
net.to(device)
# Load test data
with open(out_path+add_args_to_string('test', args, ["encoded_dim"])+'.pth', 'rb') as in_file:
    test_dataset = torch.load(in_file)

# Get the encoded representation of the test samples
encoded_samples = []
for sample in tqdm(test_dataset):
    img = sample[0][0].unsqueeze(0).to(device)
    label = sample[0][1]
    # Encode image
    net.eval()
    with torch.no_grad():
        encoded_img  = net.sample(*net.encode(img))
    # Append to list
    encoded_samples.append((encoded_img.flatten().cpu().numpy(), label))

# Visualize encoded space
color_map = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf'
        }

# Randomly sample 1k points to plot
# encoded_samples_reduced = random.sample(encoded_samples, 1000)
# Create a numpy array with sampled points
encoded_samples_np = np.vstack([x[0] for x in encoded_samples])
print("Original shape: ", encoded_samples_np.shape)

# Reduce dimensionality with Principal Components Analysis
if args.encoded_dim > 2:
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(encoded_samples_np)
    print("Compressed shape: ", principalComponents.shape)
else:
    principalComponents = encoded_samples_np

plt.figure(figsize=(8,6))
for i in tqdm(range(principalComponents.shape[0])):
    label = encoded_samples[i][1]
    components = principalComponents[i]
    plt.plot(components[0], components[1], marker='.', color=color_map[label])
plt.grid(True)
legend = [plt.Line2D([0], [0], ls='', marker='.', color=c, label=l) for l, c in color_map.items()]
plt.legend(legend, color_map.keys(), loc='lower center', bbox_to_anchor=(0.5, 1), ncol=5, fontsize=12, frameon=False)
plt.tight_layout()
plt.savefig(plots_dir+add_args_to_string("PCA_", args)+".png", transparent=True, dpi=300)
plt.show()

if args.encoded_dim == 2:
    n_images = 12

    # Two points in the encoded space corresponding to digits
    start = np.array([10, -15])
    stop = np.array([0, 10])

    # Generate samples
    line = np.linspace(start, stop, n_images)
    encoded_value_list = [torch.tensor(x).float().unsqueeze(0).to(device) for x in line]

    # Generate corresponding images
    net.eval()
    new_img_list = []
    with torch.no_grad():
        for encoded_value in encoded_value_list:
            new_img_list.append(net.decode(encoded_value))

    # Plot results
    fig, axs = plt.subplots(2, 6, figsize=(18,6))
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(new_img_list[i].squeeze().cpu().numpy(), cmap='gist_gray')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(plots_dir+"generated_digits.png", transparent=True, dpi=300)
    plt.show()
