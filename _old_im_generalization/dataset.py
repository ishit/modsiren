from utils import *
import matplotlib.pyplot as plt
import scipy.ndimage
from torch.utils.data import DataLoader, Dataset
import skimage
from PIL import Image, ImageOps
from PIL import ImageFilter
from torchvision.transforms.functional import rotate
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop, RandomCrop, Pad
import torch.nn.init as init
import torch
from hparams import hparams
import numpy as np
import os
import numpy_indexed as npi

os.environ['PYOPENGL_PLATFORM'] = 'egl'
np.random.seed(42)

def get_mgrid_2(sidelen, dim=2):
    pixel_edges = torch.linspace(-1, 1, sidelen + 1)[:-1]
    pixel_width = pixel_edges[1] - pixel_edges[0]
    pixel_centers = pixel_edges + pixel_width / 2
    tensors = tuple(dim * [pixel_centers])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape((-1, 2))
    return mgrid

class celebAGeneralization(Dataset):
    def __init__(self, filenames, num_images, imsize, patch_res, channels, mode='train'):
        super().__init__()
        with open(filenames, 'r') as f:
            self.image_path = f.readlines()

        self.num_images = min(num_images, len(self.image_path))
        self.image_path = self.image_path[:self.num_images]
        self.cell_width = patch_res
        self.imsize = imsize
        self.channels = channels
        self.mode = mode

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.get_train(idx)
        else:
            return self.get_val(idx)

    def get_train(self, idx):
        idx = idx % self.num_images
        self.transform = Compose([
            RandomCrop(self.cell_width, pad_if_needed=True, padding_mode='reflect'),
        ])

        if self.channels == 1:
            img = Image.open(self.image_path[idx].strip()).convert('L')
            img = self.transform(img)
            img = np.asarray(img) / 255
            img = np.expand_dims(img, axis=2)
        else:
            img = Image.open(self.image_path[idx].strip())
            img = self.transform(img)
            img = np.asarray(img)[:, :, :self.channels] / 255

        self.img = img
        self.img = np.transpose(self.img, (2, 0, 1))
        coords = get_mgrid_2(self.cell_width).unsqueeze(0).float()
        self.coords = coords

        return {'idx': idx, 'img': self.img, 'coords': self.coords[0]}

    def get_val(self, idx):
        self.transform = Compose([
            Resize(self.imsize),
            CenterCrop(self.imsize)
        ])

        self.expand = Compose([Pad(self.cell_width // 4, padding_mode = 'reflect')])
        if self.channels == 1:
            img = Image.open(self.image_path[idx].strip()).convert('L')
            img = self.transform(img)
            # img = ImageOps.expand(img, border = self.cell_width//4)
            img = self.expand(img)
            img = np.asarray(img) / 255
            img = np.expand_dims(img, axis=2)
        else:
            img = Image.open(self.image_path[idx].strip())
            img = self.transform(img)
            img = self.expand(img)
            # img = ImageOps.expand(img, border = self.cell_width//4)
            img = np.asarray(img)[:, :, :3] / 255

        self.img = img
        self.patches = self.get_patches(img)
        self.patches = np.transpose(self.patches, (0, 3, 1, 2))
        self.n_patches = self.patches.shape[0]

        coords = get_mgrid_2(self.cell_width).unsqueeze(0).float()
        coords = coords.repeat(self.n_patches, 1, 1)
        self.coords = coords

        return {'idx': idx, 'img': self.patches, 'coords': self.coords}

    def get_patches(self, img):
        h = self.imsize + self.cell_width
        w = self.imsize + self.cell_width

        all_patches = []
        for i in range(0, h - self.cell_width, self.cell_width // 2):
            for j in range(0, w - self.cell_width, self.cell_width // 2):
                all_patches.append(img[i:i + self.cell_width, j:j + self.cell_width])
        all_patches = np.stack(all_patches)
        return all_patches 
