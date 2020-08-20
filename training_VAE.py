import os
import torch
import argparse
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import add_args_to_string
from dataset import DoubleMnistDataset, AddGaussianNoise, AddOcclusion

# Constants
NUM_EPOCHS = 50
ENCODED_SPACE_DIM = 6
VERBOSE = False

# Define paths
data_root_dir = '../datasets'
out_path = 'datasets/'
plots_dir = 'plots/'
params_dir = 'params/'


# Define the network architecture
class VarAutoencoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_mean = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        self.encoder_logvar = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        # Return distribution parameters
        mu, logvar = self.encode(x)
        # Sample a vector from the distribution
        x = self.sample(mu, logvar)
        # Decode the vector
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar

    def sample(self, mu, logvar):
        # Transform logvar in var
        std = logvar.mul(0.5).exp_()
        # Generate a random sample from uniform distribution
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        # Use eps to generate a random sample from normal distribution
        return eps.mul(std).add_(mu)

    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


# Training function
def train_epoch(net, dataloader, loss_fn, optimizer, verbose=VERBOSE):
    # Training
    net.train()
    for sample_batch in dataloader:
        # Extract data and move tensors to the selected device
        image_batch_in = sample_batch[0][0].to(device)
        image_batch_out = sample_batch[1][0].to(device)
        # Forward pass
        output = net(image_batch_in)
        loss = loss_fn(output, image_batch_out)
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        # Print loss
        if verbose:
            print('\t partial train loss: %f' % (loss.data))


# Testing function
def test_epoch(net, dataloader, loss_fn, optimizer):
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch_in = sample_batch[0][0].to(device)
            image_batch_out = sample_batch[1][0].to(device)
            # Forward pass
            out = net(image_batch_in)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch_out.cpu()])
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def CustomLoss():
    return None


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Create the dataset.')
    parser.add_argument('--add_noise', type=float, default=0, help='Variance of Gaussian noise added over the input image' )
    parser.add_argument('--add_occlusion', type=int, default=0, help='Number of occluded rows over the input image')
    parser.add_argument('--encoded_dim', type=int, default=ENCODED_SPACE_DIM, help='Encoded space dimension')
    # parse input arguments
    args = parser.parse_args()

    # Import data
    with open(out_path+add_args_to_string('train', args, ["encoded_dim"])+'.pth', 'rb') as in_file:
        train_dataset = torch.load(in_file)
    with open(out_path+add_args_to_string('test', args, ["encoded_dim"])+'.pth', 'rb') as in_file:
        test_dataset = torch.load(in_file)

    ### Initialize the network
    net = VarAutoencoder(encoded_space_dim=args.encoded_dim)

    ### Define dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    ### Define a loss function
    loss_fn = torch.nn.MSELoss()

    ### Define an optimizer
    lr = 1e-3 # Learning rate
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    # Move all the network parameters to the selected device
    net.to(device)

    # Vector storing training history
    epoch_loss = []
    ### Training cycle
    for epoch in range(NUM_EPOCHS):
        print('EPOCH %d/%d' % (epoch + 1, NUM_EPOCHS))
        ### Training
        train_epoch(net, dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optim)
        ### Validation
        val_loss = test_epoch(net, dataloader=test_dataloader, loss_fn=loss_fn, optimizer=optim)
        epoch_loss.append(val_loss.cpu().numpy())
        # Print Validationloss
        print('\n\n\t VALIDATION - EPOCH %d/%d - loss: %f\n\n' % (epoch + 1, NUM_EPOCHS, val_loss))

        ### Plot progress
        img_noisy = test_dataset[1][0][0].unsqueeze(0).to(device)
        img_original = test_dataset[1][1][0].unsqueeze(0).to(device)
        net.eval()
        with torch.no_grad():
            rec_img  = net(img_noisy)
        fig, axs = plt.subplots(1, 3, figsize=(15,5))
        axs[0].imshow(img_original.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[0].set_title('Original input image')
        axs[1].imshow(img_noisy.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title('Noisy input image')
        axs[2].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[2].set_title('Reconstructed image (EPOCH %d)' % (epoch + 1))
        plt.tight_layout()
        # Save figures
        os.makedirs('autoencoder_progress_%d_features' % args.encoded_dim, exist_ok=True)
        plt.savefig('autoencoder_progress_%d_features/epoch_%d.png' % (args.encoded_dim, epoch + 1))
        plt.close()
    # Save network parameters
    torch.save(net.state_dict(), params_dir+'net_params_{}.pth'.format(args.encoded_dim))

    with open("loss_history.txt", "a") as infile:
        infile.write("Latent Space Dimension = {}\n".format(args.encoded_dim))
        infile.write(", ".join(str(x) for x in epoch_loss))
        infile.write("\n\n")

    print("Network successfully trained")
