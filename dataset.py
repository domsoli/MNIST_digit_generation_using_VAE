import os
import dill
import torch
import random
import argparse
import matplotlib.pyplot as plt

from tqdm import tqdm

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets import MNIST

# Define paths
DATA_ROOT_DIR = '../datasets'
OUT_PATH = 'datasets/'
PLOTS_DIR = 'plots/'


def add_args_to_string(string, args, exceptions=[]):
    """
    Append to a string the list of arguments in args with their values.
    """

    for arg in vars(args):
        if arg not in exceptions:
            string += '_{}={}'.format(arg, getattr(args, arg)) if getattr(args, arg) else ''
    return string


class AddGaussianNoise(object):

    def __init__(self, std=0.1):
        super().__init__()
        self.std = std

    def __call__(self, tensor):
        if self.std:
            tensor += torch.abs(torch.randn(tensor.size())) * self.std
        return tensor

    def __repr__(self):
        return self.__class__.__name__ +'(std={})'.format(self.std)


class AddOcclusion(object):

    def __init__(self, n_lines=0):
        super().__init__()
        self.n_lines = n_lines

    def __call__(self, tensor):
        if self.n_lines:
            lines = range((tensor.size()[1]-self.n_lines)//2,
                          (tensor.size()[1]+self.n_lines)//2)
            tensor[:, lines, :] = 0
        return tensor

    def __repr__(self):
        return self.__class__.__name__ +'(n_lines={})'.format(self.n_lines)


class DoubleMnistDataset(Dataset):
    """
    Create MNIST dataset with different transformations for input and output.
    """

    def __init__(self, data_path, train, transform_in, transform_out):
        self.data_in = MNIST(data_path, train,  download=True, transform=transform_in)
        self.data_out = MNIST(data_path, train,  download=True, transform=transform_out)

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, index):
        return self.data_in.__getitem__(index), self.data_out.__getitem__(index)

    def show_sample(self, dims, save=False):
        fig, axs = plt.subplots(*dims, figsize=(8,8))
        for ax in axs.flatten():
            img, label = random.choice(self.data_in)
            ax.imshow(img.squeeze().numpy(), cmap='gist_gray', origin="upper")
            ax.set_title('Label: %d' % label)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(PLOTS_DIR+add_args_to_string('dataset')+'.png')


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(description='Create the dataset.')
    parser.add_argument('--add_noise', type=float, default=0, help='Variance of Gaussian noise added over the input image' )
    parser.add_argument('--add_occlusion', type=int, default=0, help='Number of occluded rows over the input image')
    parser.add_argument('--show', action='store_true', help='Show some sample of the input dataset')
    # parse input arguments
    args = parser.parse_args()

    # Define transformations
    lossy_transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(args.add_noise),
        AddOcclusion(args.add_occlusion),
    ])

    base_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Define datasets
    dataset = MNIST(DATA_ROOT_DIR, download=True)
    train_dataset = DoubleMnistDataset(DATA_ROOT_DIR, True, lossy_transform, base_transform)
    test_dataset = DoubleMnistDataset(DATA_ROOT_DIR, False, lossy_transform, base_transform)

    # Plot some sample from the data
    train_dataset.show_sample((3,3), save=False)

    # Save datasets
    with open(OUT_PATH+add_args_to_string('train', args, exceptions=["show"])+'.pth', 'wb') as file:
        torch.save(train_dataset, file)
    with open(OUT_PATH+add_args_to_string('test', args, exceptions=["show"])+'.pth', 'wb') as file:
        torch.save(test_dataset, file)


if __name__ == '__main__':
    main()
