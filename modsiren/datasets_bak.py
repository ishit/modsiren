import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision.transforms import CenterCrop, Compose, Pad, RandomCrop, Resize, ToTensor
import torch
import numpy as np
import numpy_indexed as npi
import os
from modsiren.utils import *
# from utils.sample_sdf3 import *
# from utils.extensions.mesh2sdf2_cuda import mesh2sdf
# from utils.toolbox import pcl_library

os.environ['PYOPENGL_PLATFORM'] = 'egl'
# np.random.seed(42)

def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape((-1, 2))
    return mgrid


def get_mgrid_2(sidelen, dim=2):
    pixel_edges = torch.linspace(-1, 1, sidelen + 1)[:-1]
    pixel_width = pixel_edges[1] - pixel_edges[0]
    pixel_centers = pixel_edges + pixel_width / 2
    tensors = tuple(dim * [pixel_centers])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape((-1, 2))
    return mgrid


def sdfmeshfun(point, mesh):
    out_ker = mesh2sdf.mesh2sdf_gpu(point.contiguous(), mesh)[0]
    return out_ker


def trimmesh(mesh_t, residual=False):
    mesh_t = mesh_t.to("cuda:0")
    valid_triangles = mesh2sdf.trimmesh_gpu(mesh_t)
    if residual:
        valid_triangles = ~valid_triangles
    mesh_t = mesh_t[valid_triangles, :, :].contiguous()
    print("[Trimmesh] {} -> {}".format(valid_triangles.size(0), mesh_t.size(0)))
    return mesh_t


class ExtendedSingleImagePatches(Dataset):
    def __init__(self, impath, imsize=1024, cells=64, channels=1):
        super().__init__()

        self.imsize = imsize
        self.cells = cells
        self.transform = Compose([
            Resize(imsize),
            CenterCrop(imsize)
        ])

        self.larger_transform = Compose([
            Resize(imsize * 2),
            CenterCrop(imsize * 2)
        ])

        self.toTensor = Compose([ToTensor()])

        self.cell_width = self.imsize // self.cells
        self.pixel_width = 2 / self.cell_width

        if channels == 1:
            img = Image.open(impath).convert('L')
            img = self.transform(img)
            img = ImageOps.expand(img, border=self.cell_width//4)
            img = np.asarray(img) / 255
            img = np.expand_dims(img, axis=2)

        else:
            img = Image.open(impath)
            large_img = self.larger_transform(img)
            img = self.transform(img)
            img = ImageOps.expand(img, border=self.cell_width//4)
            img = np.asarray(img)[:, :, :3] / 255

            large_img = ImageOps.expand(large_img, border=self.cell_width//2)
            large_img = np.asarray(large_img) / 255

        self.img = img
        self.patches = self.get_patches(img, self.imsize, self.cell_width)
        self.patches = np.transpose(self.patches, (0, 3, 1, 2))

        self.large_img = large_img
        self.large_patches = self.get_patches(
            large_img, self.imsize * 2, self.cell_width * 2)
        self.large_patches = np.transpose(self.large_patches, (0, 3, 1, 2))

        self.coords = get_mgrid_2(self.cell_width).float()
        self.n_patches = self.patches.shape[0]

        self.large_coords = get_mgrid_2(self.cell_width * 2).float()

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        return {'idx': idx, 'img': self.patches[idx], 'coords': self.coords,
                'large_img': self.large_patches[idx], 'large_coords': self.large_coords}

    def get_patches(self, img, imsize, cell_width):
        h = imsize + cell_width
        w = imsize + cell_width

        all_patches = []
        for i in range(0, h - cell_width, cell_width // 2):
            for j in range(0, w - cell_width, cell_width // 2):
                all_patches.append(
                    img[i:i + cell_width, j:j + cell_width])
        all_patches = np.stack(all_patches)
        return all_patches


class SingleImagePatches(Dataset):
    def __init__(self, impath, imsize=1024, cells=64, channels=1):
        super().__init__()

        self.imsize = imsize
        self.cells = cells
        self.transform = Compose([
            Resize(imsize),
            CenterCrop(imsize)
        ])
        self.larger_transform = Compose([
            Resize(imsize * 2),
            CenterCrop(imsize * 2)
        ])
        self.toTensor = Compose([ToTensor()])

        self.cell_width = self.imsize // self.cells
        self.pixel_width = 2 / self.cell_width

        if channels == 1:
            img = Image.open(impath).convert('L')
            img = self.transform(img)
            img = np.asarray(img) / 255
        else:
            img = Image.open(impath)
            large_img = self.larger_transform(img)
            img = self.transform(img)
            img = np.asarray(img)[:, :, :3] / 255

            # large_img = ImageOps.expand(large_img, border=self.cell_width//2)
            large_img = np.asarray(large_img) / 255

        self.img = img.astype(np.float32)

        if channels == 1:
            patches = blockshaped(img, self.cell_width, self.cell_width)
            self.patches = np.expand_dims(patches, -3)
        else:
            patches = blockshaped_wc(img, self.cell_width, self.cell_width)
            self.patches = np.transpose(patches, (0, 3, 1, 2))
            large_patches = blockshaped_wc(
                large_img, self.cell_width * 2, self.cell_width * 2)
            self.large_patches = np.transpose(large_patches, (0, 3, 1, 2))

        self.coords = get_mgrid_2(self.cell_width).float()
        self.large_coords = get_mgrid_2(self.cell_width * 2).float()

        tensors = tuple(2 * [np.linspace(-1, 1, num=imsize)])

        xx, yy = np.meshgrid(*tensors)
        mgrid = np.stack([yy, xx], axis=-1)
        patches = blockshaped_wc(mgrid, self.cell_width, self.cell_width)
        self.global_coords = patches.reshape((-1, self.cell_width ** 2, 2))
        self.global_coords = torch.FloatTensor(self.global_coords)
        self.n_patches = self.patches.shape[0]

    def __len__(self):
        return self.n_patches * 50

    def __getitem__(self, idx):
        idx = idx % self.n_patches
        return {'idx': idx, 'img': self.patches[idx], 'coords': self.coords,
                'global_coords': self.global_coords[idx], 'large_img': self.large_patches[idx],
                'large_coords': self.large_coords}


class MultiImagePatches(Dataset):
    def __init__(self, root, filenames, num_images=1, imsize=1024, cells=64, channels=1):
        super().__init__()

        with open(filenames, 'r') as f:
            self.image_path = f.readlines()
        self.root = root
        self.num_images = min(num_images, len(self.image_path))

        self.image_path = self.image_path[:self.num_images]
        self.tiles_per_image = 4

        self.imsize = imsize
        self.cells = cells
        self.transform = Compose([
            Resize(imsize),
            CenterCrop(imsize)
        ])
        self.toTensor = Compose([ToTensor()])
        self.channels = channels

        self.cell_width = self.imsize // self.cells
        self.pixel_width = 2 / self.cell_width
        self.coords = get_mgrid_2(self.cell_width).float()

        self.all_patches = []
        for im_id in range(self.num_images):
            impath = os.path.join(self.root, self.image_path[im_id].strip())

            if self.channels == 1:
                img = Image.open(impath).convert('L')
                img = self.transform(img)
                img = np.asarray(img) / 255
            else:
                img = Image.open(impath)
                img = self.transform(img)
                img = np.asarray(img)[:, :, :3] / 255

            self.img = img.astype(np.float32)

            if self.channels == 1:
                patches = blockshaped(img, self.cell_width, self.cell_width)
                patches = np.expand_dims(patches, -3)
            else:
                patches = blockshaped_wc(img, self.cell_width, self.cell_width)
                patches = np.transpose(patches, (0, 3, 1, 2))

            self.all_patches.append(patches)

    def __len__(self):
        return self.num_images * self.all_patches[0].shape[0]

    def __getitem__(self, idx):
        im_id = np.random.randint(0, self.num_images)
        patch_id = np.random.randint(0, self.all_patches[0].shape[0])

        return {'idx': idx, 'img': self.all_patches[im_id][patch_id], 'coords': self.coords}


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
            RandomCrop(self.cell_width, pad_if_needed=True,
                       padding_mode='reflect'),
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

        self.expand = Compose(
            [Pad(self.cell_width // 4, padding_mode='reflect')])
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
                all_patches.append(
                    img[i:i + self.cell_width, j:j + self.cell_width])
        all_patches = np.stack(all_patches)
        return all_patches


class Video(Dataset):
    def __init__(self, path_to_video, channels, cells, mode='local'):
        super().__init__()
        self.vid = np.load(path_to_video)
        self.vid = self.vid[:, :, :, :channels]

        # Reshape video for 256 x 640
        self.vid = self.vid[:, 8:-8, :, :]

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]
        self.n_voxels = list(cells)
        self.mode = mode

        self.cell_size = [self.shape[0] // self.n_voxels[0],
                          self.shape[1] // self.n_voxels[1],
                          self.shape[2] // self.n_voxels[2]]

        # if not os.path.exists('data/extended_bikes.npy'):
        # self.extended_vid = self.get_extended_video(self.vid)
        # self.vid = video_unblockshaped_wc(self.extended_vid, 2 * self.shape[1] - self.cell_size[1], 2 * self.shape[2] - self.cell_size[2])
        # np.save('data/extended_bikes.npy', self.vid)
        # else:
        # self.vid = np.load('data/extended_bikes.npy')
        # self.shape = self.vid.shape[:-1]

        # self.n_voxels[1] = self.n_voxels[1] * 2 - 1
        # self.n_voxels[2] = self.n_voxels[2] * 2 - 1

        self.local_coords = self.get_vid_coords(self.cell_size)
        self.global_coords = self.get_vid_coords(self.shape)
        self.global_coords = self.global_coords.view(*self.vid.shape[:-1], 3)
        self.data = torch.from_numpy(self.vid)

    def __len__(self):
        return np.product(self.n_voxels)

    def __getitem__(self, idx):
        # 1D to 3D index
        xi = idx % self.n_voxels[0]
        yi = (idx // self.n_voxels[0]) % self.n_voxels[1]
        zi = ((idx // self.n_voxels[0]) // self.n_voxels[1]) % self.n_voxels[2]

        len_x = self.cell_size[0]
        len_y = self.cell_size[1]
        len_z = self.cell_size[2]

        data = self.data[xi * len_x: (xi + 1) * len_x,
                         yi * len_y: (yi + 1) * len_y,
                         zi * len_z: (zi + 1) * len_z]

        coords = self.global_coords[xi * len_x: (xi + 1) * len_x,
                                    yi * len_y: (yi + 1) * len_y,
                                    zi * len_z: (zi + 1) * len_z]
        coords = coords.contiguous().view(-1, 3)

        in_dict = {'idx': idx, 'coords': self.local_coords,
                   'global_coords': coords}
        gt_dict = {'img': data}

        return in_dict, gt_dict

    def get_vid_coords(self, cell_size):
        pixel_coords = np.stack(
            np.mgrid[:cell_size[0], :cell_size[1], :cell_size[2]],
            axis=-1)[None, ...].astype(np.float32)

        pixel_coords = pixel_coords.reshape((-1, 3))
        pixel_coords[:, 0] = pixel_coords[:, 0] / (cell_size[0] - 1)
        pixel_coords[:, 1] = pixel_coords[:, 1] / (cell_size[1] - 1)
        pixel_coords[:, 2] = pixel_coords[:, 2] / (cell_size[2] - 1)
        pixel_coords = pixel_coords * 2 - 1
        pixel_coords = torch.Tensor(pixel_coords).view(-1, 3)
        return pixel_coords

    def get_extended_video(self, vid):
        h = self.shape[1]
        w = self.shape[2]
        all_patches = []
        for i in range(0, h - self.cell_size[1] + 1, self.cell_size[1] // 2):
            for j in range(0, w - self.cell_size[2] + 1, self.cell_size[2] // 2):
                all_patches.append(
                    vid[:, i:i + self.cell_size[1], j:j + self.cell_size[2]])

        all_patches = np.stack(all_patches)
        return all_patches


class SingleVideoDataset(Dataset):
    def __init__(self, path_to_video, channels, cell_size, mode='local'):
        super().__init__()
        self.vid = np.load(path_to_video).astype(np.float32)
        self.vid = self.vid[:, :, :, :channels]
        old_shape = self.vid.shape

        new_t = old_shape[0] - old_shape[0] % cell_size[0]
        new_h = old_shape[1] - old_shape[1] % cell_size[1]
        new_w = old_shape[2] - old_shape[2] % cell_size[2]

        self.vid = self.vid[:new_t, :new_h, :new_w]

        self.shape = self.vid.shape[:-1]
        self.channels = self.vid.shape[-1]

        self.n_voxels = [self.shape[i] // cell_size[i] for i in range(3)]
        self.mode = mode
        self.cell_size = cell_size

        self.local_coords = self.get_vid_coords(self.cell_size)
        self.global_coords = self.get_vid_coords(self.shape)
        self.global_coords = self.global_coords.view(*self.vid.shape[:-1], 3)
        self.data = torch.from_numpy(self.vid)

    def __len__(self):
        return np.product(self.n_voxels)

    def __getitem__(self, idx):
        # 1D to 3D index
        xi = idx % self.n_voxels[0]
        yi = (idx // self.n_voxels[0]) % self.n_voxels[1]
        zi = ((idx // self.n_voxels[0]) // self.n_voxels[1]) % self.n_voxels[2]

        len_x = self.cell_size[0]
        len_y = self.cell_size[1]
        len_z = self.cell_size[2]

        data = self.data[xi * len_x: (xi + 1) * len_x,
                         yi * len_y: (yi + 1) * len_y,
                         zi * len_z: (zi + 1) * len_z]

        coords = self.global_coords[xi * len_x: (xi + 1) * len_x,
                                    yi * len_y: (yi + 1) * len_y,
                                    zi * len_z: (zi + 1) * len_z]
        coords = coords.contiguous().view(-1, 3)

        return {'img': data, 'idx': idx, 'coords': self.local_coords, 'global_coords': coords}

    def get_vid_coords(self, cell_size):
        pixel_coords = np.stack(
            np.mgrid[:cell_size[0], :cell_size[1], :cell_size[2]],
            axis=-1)[None, ...].astype(np.float32)

        pixel_coords = pixel_coords.reshape((-1, 3))
        pixel_coords[:, 0] = pixel_coords[:, 0] / (cell_size[0] - 1)
        pixel_coords[:, 1] = pixel_coords[:, 1] / (cell_size[1] - 1)
        pixel_coords[:, 2] = pixel_coords[:, 2] / (cell_size[2] - 1)
        pixel_coords = pixel_coords * 2 - 1
        pixel_coords = torch.Tensor(pixel_coords).view(-1, 3)
        return pixel_coords

    def get_extended_video(self, vid):
        h = self.shape[1]
        w = self.shape[2]
        all_patches = []
        for i in range(0, h - self.cell_size[1] + 1, self.cell_size[1] // 2):
            for j in range(0, w - self.cell_size[2] + 1, self.cell_size[2] // 2):
                all_patches.append(
                    vid[:, i:i + self.cell_size[1], j:j + self.cell_size[2]])

        all_patches = np.stack(all_patches)
        return all_patches

class MultiVideoDataset(Dataset):
    def __init__(self, path_to_dir, channels, cell_size, mode='local'):
        self.vid_datasets = []
        for f in os.listdir(path_to_dir):
            vid_path = os.path.join(path_to_dir, f)
            print(vid_path)
            d = SingleVideoDataset(vid_path, channels, cell_size) 
            self.vid_datasets.append(d)
        
        self.len = sum([len(d) for d in self.vid_datasets])
        self.n_vids = len(self.vid_datasets)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        rand_vid_idx = np.random.randint(0, self.n_vids) 
        rand_voxel_idx = np.random.randint(0, len(self.vid_datasets[rand_vid_idx]))
        return self.vid_datasets[rand_vid_idx][rand_voxel_idx]
                

class VideoDataset(Dataset):
    def __init__(self, root, filenames, num_videos, cell_size, crop_size, random_crop=True):
        super().__init__()
        with open(filenames, 'r') as f:
            self.sequences = f.readlines()[:num_videos]

        self.cell_size = cell_size
        self.crop_size = crop_size
        self.root = root

        self.local_coords = self.get_vid_coords(self.cell_size)
        self.local_coords_2x = self.get_vid_coords(
            [self.cell_size[0], self.cell_size[1] * 2, self.cell_size[2] * 2])

        if random_crop:
            self.crop_transform = RandomCrop(
                self.crop_size, pad_if_needed=True, padding_mode='reflect')
        else:
            self.crop_transform = CenterCrop(self.crop_size)

        self.resize_transform = Resize(self.cell_size[1])
        self.resize_transform_2x = Resize(self.cell_size[1] * 2)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        idx = idx % len(self.sequences)
        frames = []
        frames_2x = []

        for i in range(7):
            vid = self.sequences[idx].strip()
            vid_dir = os.path.join(self.root, vid)
            img = Image.open(os.path.join(vid_dir, f'im{i+1}.png'))
            img = self.crop_transform(img)
            img_2x = self.resize_transform_2x(img)
            img = self.resize_transform(img)
            img = np.asarray(img) / 255.
            img_2x = np.array(img_2x) / 255.

            frames.append(img)
            frames_2x.append(img_2x)

        data = np.stack(frames)
        data = np.transpose(data, (3, 0, 1, 2))

        data_2x = np.stack(frames_2x)
        data_2x = np.transpose(data_2x, (3, 0, 1, 2))

        return {'idx': idx, 'img': data, 'img_2x': data_2x, 'coords': self.local_coords,
                'coords_2x': self.local_coords_2x}

    def get_vid_coords(self, cell_size):
        pixel_coords = np.stack(
            np.mgrid[:cell_size[0], :cell_size[1], :cell_size[2]],
            axis=-1)[None, ...].astype(np.float32)

        pixel_coords = pixel_coords.reshape((-1, 3))
        pixel_coords[:, 0] = pixel_coords[:, 0] / (cell_size[0] - 1)
        pixel_coords[:, 1] = pixel_coords[:, 1] / (cell_size[1] - 1)
        pixel_coords[:, 2] = pixel_coords[:, 2] / (cell_size[2] - 1)
        pixel_coords = pixel_coords * 2 - 1
        pixel_coords = torch.Tensor(pixel_coords).view(-1, 3)
        return pixel_coords


class DiligentDataset(Dataset):
    def __init__(self, imsize=1024):
        super().__init__()
        self.imsize = imsize
        self.transform = Compose([
            Resize(self.imsize),
            CenterCrop(self.imsize)
        ])
        self.toTensor = Compose([ToTensor()])
        self.imgs = []
        light_directions = []
        scale_factor = 4.0

        with open('data/light_directions.txt', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                img_path = os.path.join(
                    'data/processed_pot_color', f'{i:03d}.png')
                img = Image.open(img_path)
                img = self.transform(img)
                img = np.asarray(img) / 255
                self.imgs.append(img * scale_factor)
                light_directions.append(line.strip().split(' '))

        self.light_directions = np.array(light_directions).astype(np.float32)
        self.coords = get_mgrid_2(self.imsize)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return {'idx': idx, 'img': self.imgs[idx], 'coords': self.coords, 'embedding': self.light_directions[idx]}


class TexturesDataset(Dataset):
    def __init__(self, root, filenames, num_images, imsize, channels, crop_size, random_crop=True):
        super().__init__()
        with open(filenames, 'r') as f:
            self.image_path = f.readlines()

        self.root = root
        self.num_images = min(num_images, len(self.image_path))
        # self.train_codes = torch.randn(self.num_images, 256) * 0.001

        self.image_path = self.image_path[:self.num_images]
        self.imsize = imsize
        self.channels = channels
        self.coords = get_mgrid_2(self.imsize).float()
        # self.coords = get_mgrid(self.imsize).float()
        self.coords_2x = get_mgrid_2(2 * self.imsize).float()

        self.crop_size = crop_size

        if random_crop:
            self.crop_transform = RandomCrop(
                self.crop_size, pad_if_needed=True, padding_mode='reflect')
        else:
            self.crop_transform = CenterCrop(self.crop_size)

        self.resize_transform = Resize(self.imsize)
        self.resize_transform_2x = Resize(2 * self.imsize)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        filename = os.path.join(self.root, self.image_path[idx].strip())
        if self.channels == 1:
            img = Image.open(filename).convert('L')
            img = self.crop_transform(img)
            img_2x = self.resize_transform_2x(img)
            img = self.resize_transform(img)

            img = np.asarray(img) / 255
            img = np.expand_dims(img, axis=2)

            img_2x = np.asarray(img_2x) / 255
            img_2x = np.expand_dims(img_2x, axis=2)
        else:
            img = Image.open(filename)
            img = self.crop_transform(img)
            img_2x = self.resize_transform_2x(img)
            img = self.resize_transform(img)

            img = np.asarray(img)[:, :, :self.channels] / 255
            img_2x = np.asarray(img_2x)[:, :, :self.channels] / 255

        img = np.transpose(img, (2, 0, 1))
        img_2x = np.transpose(img_2x, (2, 0, 1))

        return {'idx': idx, 'img': img, 'img_2x': img_2x, 'coords': self.coords, 'coords_2x': self.coords_2x}
        # 'latent': self.train_codes[idx]}


class SuperRes(Dataset):
    def __init__(self, root, filenames, num_images, imsize, channels, crop_size, random_crop=True):
        super().__init__()
        with open(filenames, 'r') as f:
            self.image_path = f.readlines()

        self.root = root
        self.num_images = min(num_images, len(self.image_path))
        self.image_path = self.image_path[:self.num_images]
        self.imsize = imsize
        self.channels = channels
        self.coords = get_mgrid_2(self.imsize).float()
        self.crop_size = crop_size

        if random_crop:
            self.crop_transform = RandomCrop(
                self.crop_size, pad_if_needed=True, padding_mode='reflect')
        else:
            self.crop_transform = CenterCrop(self.crop_size)

        self.resize_transform = Resize(self.imsize)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        filename = os.path.join(self.root, self.image_path[idx].strip())
        if self.channels == 1:
            img = Image.open(filename).convert('L')
            img = self.crop_transform(img)
            img = self.resize_transform(img)

            img = np.asarray(img) / 255
            img = np.expand_dims(img, axis=2)
        else:
            img = Image.open(filename)
            img = self.crop_transform(img)
            img = self.resize_transform(img)
            img = np.asarray(img)[:, :, :self.channels] / 255

        img = np.transpose(img, (2, 0, 1))

        return {'idx': idx, 'img': img, 'coords': self.coords}


class ShapeDataset(Dataset):
    def __init__(self, root, filenames, num_shapes, n_voxels, sample_size):
        with open(filenames, 'r') as f:
            self.shape_list = f.readlines()

        self.root = root
        self.num_shapes = min(num_shapes, len(self.shape_list))
        self.n_voxels = n_voxels
        self.sample_size = sample_size
        self.surface_samples = (sample_size * 9) // 10
        self.sphere_samples = sample_size - self.surface_samples

    def __len__(self):
        return self.num_shapes

    def __getitem__(self, idx):
        file_path_surface = os.path.join(
            self.root, 'surface', self.shape_list[idx].strip())
        points_surface_sdf = np.load(file_path_surface)
        if points_surface_sdf.shape[0] < 1:
            idx = (idx + 1) % self.num_shapes  # skip
            return self.__getitem__(idx)

        file_path_sphere = os.path.join(
            self.root, 'sphere', self.shape_list[idx].strip())
        points_sphere_sdf = np.load(file_path_sphere)

        random_idcs_sphere = np.random.choice(
            points_sphere_sdf.shape[0], self.sphere_samples)
        random_idcs_surface = np.random.choice(
            points_surface_sdf.shape[0], self.surface_samples)

        coords = np.concatenate([points_sphere_sdf[random_idcs_sphere, :3],
                                 points_surface_sdf[random_idcs_surface, :3]], axis=0)
        sdf = np.concatenate([points_sphere_sdf[random_idcs_sphere, 3:4],
                              points_surface_sdf[random_idcs_surface, 3:4]], axis=0)

        T_coords, voxel_idx = self.voxelize_points(coords, self.n_voxels)
        shape_idx = np.ones(voxel_idx.shape) * idx

        return {'coords': T_coords, 'shape_idx': shape_idx, 'voxel_idx': voxel_idx, 'sdf': sdf}

    def voxelize_points(self, coords, n_voxels):
        bins = np.linspace(-1, 1, n_voxels + 1)
        bin_idx = np.searchsorted(bins, coords) - 1
        bin_idx = np.clip(bin_idx, 0, n_voxels - 1)
        voxel_idx = bin_idx[..., 0] * n_voxels * n_voxels +\
            bin_idx[..., 1] * n_voxels + bin_idx[..., 2]
        T_coords = coords - bins[bin_idx]
        sidelen = bins[1] - bins[0]
        T_coords = T_coords / sidelen
        T_coords = T_coords - 0.5
        T_coords = 2 * T_coords

        return T_coords, voxel_idx


class DragonDataset(Dataset):
    def __init__(self, n_voxels, points_path, sdf_path, sample_size=512*512, extended_field=False, n_points=-1, mode='modsiren'):
        super().__init__()
        self.n_voxels = n_voxels
        self.sample_size = sample_size
        self.bins = torch.linspace(-1, 1, self.n_voxels + 1)
        self.extended_field = False
        self.mode = mode

        points = np.load(points_path)
        sdf = np.load(sdf_path)

        print(points.shape)
        if self.mode == 'modsiren':
            T_points, latent_idx = self.voxelize_points(points, n_voxels)

            # Eliminate points in empty voxels
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            ldx = latent_idx
            s = sdf

            g = npi.group_by(ldx[:, 0])
            b = g.min(np.abs(sdf))

            non_empty_voxel_idx = b[0][b[1] < 0.1]
            mask = np.isin(ldx[:, 0], non_empty_voxel_idx)

            T_points = T_points[mask]
            s = s[mask]
            ldx = ldx[mask]
            all_pts = points[mask]

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            latent_idx = ldx
            sdf = torch.FloatTensor(s)

            self.non_empty_voxel_idx = non_empty_voxel_idx
            random_idx = np.random.permutation(sdf.shape[0])[:n_points]
            self.all_points = all_pts[random_idx]
            self.all_T_points = T_points[random_idx]
            self.all_sdf = sdf[random_idx]
            self.all_latent_idx = latent_idx[random_idx]

        else:
            random_idx = np.random.permutation(sdf.shape[0])[:n_points]
            self.all_points = points[random_idx]
            self.all_T_points = points[random_idx]
            self.all_sdf = sdf[random_idx]
            self.all_latent_idx = np.zeros((self.all_points.shape[0], 1))

        self.len = self.all_points.shape[0] // self.sample_size

    def __len__(self):
        return self.len * 5

    def __getitem__(self, idx):
        mask = np.random.choice(self.all_points.shape[0], self.sample_size)
        coords = self.all_points[mask]
        T_coords = self.all_T_points[mask]
        latent_idx = self.all_latent_idx[mask]
        sdf = self.all_sdf[mask]

        if self.extended_field:
            T_coords = T_coords / 1.5

        return {'coords': T_coords,
                'global_coords': coords,
                'voxel_idx': latent_idx,
                'sdf': sdf}

    def voxelize_points(self, coords, n_voxels):
        bins = np.linspace(-1, 1, n_voxels + 1)
        bin_idx = np.searchsorted(bins, coords) - 1
        bin_idx = np.clip(bin_idx, 0, n_voxels - 1)
        latent_idx = bin_idx[..., :1] * n_voxels * n_voxels +\
            bin_idx[..., 1:2] * n_voxels + bin_idx[..., 2:]
        T_coords = coords - bins[bin_idx]
        sidelen = bins[1] - bins[0]
        T_coords = T_coords / sidelen
        T_coords = T_coords - 0.5
        T_coords = 2 * T_coords
        return T_coords, latent_idx


class DynamicDragon(Dataset):

    def __init__(self, n_voxels, mesh_path, sample_size=512*512, extended_field=False, n_points=500000, mode='modsiren'):
        super().__init__()
        self.n_voxels = n_voxels
        self.sample_size = sample_size
        self.bins = torch.linspace(-1, 1, self.n_voxels + 1)
        self.extended_field = False
        self.mode = mode
        device = torch.device('cuda')
        self.device = device
        self.mesh = self.load_mesh(mesh_path).to(device)
        self.mesh_np = self.mesh.cpu().numpy()
        self.n_points = n_points

        num_surface_samples = n_points // 2
        num_sphere_samples = n_points // 2
        self.noise_vec = torch.empty(
            [num_surface_samples, 3], dtype=torch.float32, device=device)  # x y z
        self.noise_vec2 = torch.empty(
            [num_sphere_samples, 3], dtype=torch.float32, device=device)  # x y z
        self.noise_vec3 = torch.empty(
            [num_sphere_samples, 1], dtype=torch.float32, device=device)  # x y z
        self.resample()

    def remove_empty_voxels(self, points, sdf):
        if self.mode == 'modsiren':
            T_points, latent_idx = self.voxelize_points(points, self.n_voxels)

            # Eliminate points in empty voxels
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            ldx = latent_idx
            s = sdf

            g = npi.group_by(ldx[:, 0])
            b = g.min(np.abs(sdf))

            non_empty_voxel_idx = b[0][b[1] < 0.1]
            mask = np.isin(ldx[:, 0], non_empty_voxel_idx)

            T_points = T_points[mask]
            s = s[mask]
            ldx = ldx[mask]
            all_pts = points[mask]

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            latent_idx = ldx
            sdf = torch.FloatTensor(s)

            self.non_empty_voxel_idx = non_empty_voxel_idx
            random_idx = np.random.permutation(sdf.shape[0])
            self.all_points = all_pts[random_idx]
            self.all_T_points = T_points[random_idx]
            self.all_sdf = sdf[random_idx]
            self.all_latent_idx = latent_idx[random_idx]

        else:
            random_idx = np.random.permutation(sdf.shape[0])[:n_points]
            self.all_points = points[random_idx]
            self.all_T_points = points[random_idx]
            self.all_sdf = sdf[random_idx]
            self.all_latent_idx = np.zeros((self.all_points.shape[0], 1))

    def get_points(self, sample_size):
        num_surface_samples = sample_size // 2
        num_sphere_samples = sample_size // 2
        pcl = self.pcl

        # Surface points
        # noise_vec = torch.randn(num_surface_samples, 3).to(device) * np.sqrt(0.005)
        self.noise_vec.normal_(0, np.sqrt(0.005))

        points1 = pcl + self.noise_vec
        self.noise_vec.normal_(0, np.sqrt(0.0005))
        # noise_vec = torch.randn(num_surface_samples, 3).to(device) * np.sqrt(0.005)
        points2 = pcl + self.noise_vec

        # Unit sphere points
        # for global methods
        self.noise_vec2.normal_(0, 1)
        shell_points = self.noise_vec2 / \
            torch.sqrt(torch.sum(self.noise_vec2**2, dim=-1, keepdim=True))

        self.noise_vec3.uniform_(0, 1)  # r = 1
        points3 = shell_points * (self.noise_vec3**(1/3))
        # points3 = points3.to(device)
        all_points = torch.cat([points1, points2, points3], dim=0)
        sample_dist = sdfmeshfun(all_points, self.mesh)

        return all_points.cpu().numpy(), sample_dist.cpu().numpy()

    def __len__(self):
        self.len = 100
        return self.len

    def resample(self):
        self.pcl = torch.from_numpy(pcl_library.mesh2pcl(
            self.mesh.cpu().numpy(), self.n_points // 2)).to(self.device)  # [N, 3]
        points, sdf = self.get_points(self.n_points)
        self.remove_empty_voxels(points, sdf)

    def __getitem__(self, idx):
        if idx == self.len - 1:
            self.resample()

        mask = np.random.choice(self.all_points.shape[0], self.sample_size)
        coords = self.all_points[mask]
        T_coords = self.all_T_points[mask]
        latent_idx = self.all_latent_idx[mask]
        sdf = self.all_sdf[mask]

        if self.extended_field:
            T_coords = T_coords / 1.5

        return {'coords': T_coords,
                'global_coords': coords,
                'voxel_idx': latent_idx,
                'sdf': sdf}

    def voxelize_points(self, coords, n_voxels):
        bins = np.linspace(-1, 1, n_voxels + 1)
        bin_idx = np.searchsorted(bins, coords) - 1
        bin_idx = np.clip(bin_idx, 0, n_voxels - 1)
        latent_idx = bin_idx[..., :1] * n_voxels * n_voxels +\
            bin_idx[..., 1:2] * n_voxels + bin_idx[..., 2:]
        T_coords = coords - bins[bin_idx]
        sidelen = bins[1] - bins[0]
        T_coords = T_coords / sidelen
        T_coords = T_coords - 0.5
        T_coords = 2 * T_coords
        return T_coords, latent_idx

    def load_mesh(self, mesh_path):
        mesh = trimesh.load_mesh(mesh_path)
        # mesh = as_mesh(mesh)
        mesh = mesh.vertices[mesh.faces]

        mesh[:, :, 1] *= -1
        # normalize mesh
        mesh = mesh.reshape(-1, 3)
        mesh_max = np.amax(mesh, axis=0)
        mesh_min = np.amin(mesh, axis=0)
        mesh_center = (mesh_max + mesh_min) / 2
        mesh = mesh - mesh_center
        # Find the max distance to origin
        max_dist = np.sqrt(np.max(np.sum(mesh**2, axis=-1)))
        mesh_scale = 1.0 / max_dist
        mesh *= mesh_scale
        mesh = mesh.reshape(-1, 3, 3)
        mesh_t = torch.from_numpy(mesh.astype(np.float32)).contiguous()
        return mesh_t


class SingleShapeDataset(Dataset):
    def __init__(self, n_voxels, points_path, sdf_path, sample_size=512*512, extended_field=False, n_points=-1, mode='modsiren'):
        super().__init__()
        self.n_voxels = n_voxels
        self.sample_size = sample_size
        self.bins = torch.linspace(-1, 1, self.n_voxels + 1)
        self.extended_field = False
        self.mode = mode

        points = np.load(points_path)
        sdf = np.load(sdf_path)

        print(points.shape)
        if self.mode == 'modsiren':
            T_points, latent_idx = self.voxelize_points(points, n_voxels)

            # Eliminate points in empty voxels
            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            ldx = latent_idx
            s = sdf

            g = npi.group_by(ldx[:, 0])
            b = g.min(np.abs(sdf))

            non_empty_voxel_idx = b[0][b[1] < 0.1]
            mask = np.isin(ldx[:, 0], non_empty_voxel_idx)

            T_points = T_points[mask]
            s = s[mask]
            ldx = ldx[mask]
            all_pts = points[mask]

            # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

            latent_idx = ldx
            sdf = torch.FloatTensor(s)

            self.non_empty_voxel_idx = non_empty_voxel_idx
            random_idx = np.random.permutation(sdf.shape[0])[:n_points]
            self.all_points = all_pts[random_idx]
            self.all_T_points = T_points[random_idx]
            self.all_sdf = sdf[random_idx]
            self.all_latent_idx = latent_idx[random_idx]

        else:
            random_idx = np.random.permutation(sdf.shape[0])[:n_points]
            self.all_points = points[random_idx]
            self.all_T_points = points[random_idx]
            self.all_sdf = sdf[random_idx]
            self.all_latent_idx = np.zeros((self.all_points.shape[0], 1))

        self.bins = np.linspace(-1, 1, self.n_voxels + 1)
        self.sidelen = self.bins[1] - self.bins[0]
        # self.len = self.all_points.shape[0] // self.sample_size
        self.sample_size = n_points // len(self.non_empty_voxel_idx)

    def __len__(self):
        return len(self.non_empty_voxel_idx)

    def __getitem__(self, idx):
        voxel_idx = self.unravel_idx(self.non_empty_voxel_idx[idx])

        voxel_center = np.asarray([self.bins[voxel_idx[i]]
                                   for i in range(3)]) + self.sidelen / 2
        voxel_center = voxel_center.reshape((1, 3))

        mask = np.where(np.linalg.norm(self.all_points -
                                       voxel_center, axis=1) < (self.sidelen * 0.75))
        coords = self.all_points[mask]

        random_idx = np.random.randint(0, coords.shape[0], self.sample_size)

        T_coords = (coords - voxel_center) / (self.sidelen / 2)
        latent_idx = np.ones(
            (T_coords.shape[0], 1)) * self.non_empty_voxel_idx[idx]
        sdf = self.all_sdf[mask]

        return {'coords': T_coords[random_idx],
                'global_coords': coords[random_idx],
                'voxel_idx': latent_idx[random_idx],
                'sdf': sdf[random_idx]}

    def unravel_idx(self, idx):
        z = idx % self.n_voxels
        y = (idx // self.n_voxels) % self.n_voxels
        x = idx // (self.n_voxels ** 2)

        return (x, y, z)

    def voxelize_points(self, coords, n_voxels):
        bins = np.linspace(-1, 1, n_voxels + 1)
        bin_idx = np.searchsorted(bins, coords) - 1
        bin_idx = np.clip(bin_idx, 0, n_voxels - 1)
        latent_idx = bin_idx[..., :1] * n_voxels * n_voxels +\
            bin_idx[..., 1:2] * n_voxels + bin_idx[..., 2:]
        T_coords = coords - bins[bin_idx]
        sidelen = bins[1] - bins[0]
        T_coords = T_coords / sidelen
        T_coords = T_coords - 0.5
        T_coords = 2 * T_coords
        return T_coords, latent_idx


class MultiShapeDataset(Dataset):
    def __init__(self, n_voxels, root_dir, n_shapes=2, sample_size=512*512, extended_field=False, n_points=-1, mode='modsiren'):
        super().__init__()
        self.n_voxels = n_voxels
        self.sample_size = sample_size
        self.bins = torch.linspace(-1, 1, self.n_voxels + 1)
        self.extended_field = False
        self.mode = mode
        self.n_points = n_points

        self.points_filenames = []
        self.surface_points_filenames = []
        self.sdf_filenames = []
        self.mesh_filenames = []

        for f in os.listdir(root_dir):
            if not 'ply' in f:
                continue

            self.mesh_filenames.append(os.path.join(root_dir, f))
            mesh_id = f.strip().split('.')[0]
            points_path = os.path.join(root_dir, mesh_id+'_points.npy')
            surface_points_path = os.path.join(
                root_dir, mesh_id+'_surface_points.npy')
            sdf_path = os.path.join(root_dir, mesh_id+'_sdf.npy')
            self.points_filenames.append(points_path)
            self.surface_points_filenames.append(surface_points_path)
            self.sdf_filenames.append(sdf_path)

            if len(self.points_filenames) == n_shapes:
                break


        self.mesh_directory = []
        for i in range(len(self.points_filenames)):
            print(self.points_filenames[i])
            surface_points = np.load(self.points_filenames[i])
            points = np.load(self.points_filenames[i])
            sdf = np.load(self.sdf_filenames[i])
            random_idx = np.random.randint(
                0, points.shape[0], self.sample_size)

            points = points[random_idx]
            sdf = sdf[random_idx]

            T_points, latent_idx = self.voxelize_points(points, self.n_voxels)
            T_surface_points, surface_latent_idx = self.voxelize_points(surface_points, self.n_voxels)

            self.mesh_directory.append({'off_surface': {'points': points, 'sdf': sdf, 'T_points': T_points, 'latent_idx': latent_idx},\
                    'surface': {'points': surface_points, 'sdf': 0, 'T_points': T_surface_points, 'latent_idx': surface_latent_idx},
                    # 'non_empty_voxel_idx': np.unique(surface_latent_idx)})
                    'non_empty_voxel_idx': np.unique(latent_idx)})

        self.len = sum(x['non_empty_voxel_idx'].shape[0] for x in self.mesh_directory)
        self.bins = np.linspace(-1, 1, self.n_voxels + 1)
        self.sidelen = self.bins[1] - self.bins[0]
        self.sample_size = sample_size
        self.n_shapes = len(self.mesh_directory)
        print(self.n_shapes, 'shapes')

    def get_shape_voxel_idx(self, idx):
        total_voxels = 0
        for i in range(self.n_shapes):
            # total_voxels += len(self.mesh_directory[i][-1])
            total_voxels += self.mesh_directory[i]['non_empty_voxel_idx'].shape[0]
            if idx < total_voxels:
                return i, idx - (total_voxels - self.mesh_directory[i]['non_empty_voxel_idx'].shape[0])
        return None

    def process_points(self, points, sdf):
        T_points, latent_idx = self.voxelize_points(points, self.n_voxels)

        # Eliminate points in empty voxels
        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        ldx = latent_idx
        s = sdf

        g = npi.group_by(ldx[:, 0])
        b = g.min(np.abs(sdf))

        non_empty_voxel_idx = b[0][b[1] < 0.1]
        mask = np.isin(ldx[:, 0], non_empty_voxel_idx)

        T_points = T_points[mask]
        s = s[mask]
        ldx = ldx[mask]
        all_pts = points[mask]

        # xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

        latent_idx = ldx
        sdf = torch.FloatTensor(s)

        random_idx = np.random.permutation(sdf.shape[0])[:self.n_points]
        all_points = all_pts[random_idx]
        all_sdf = sdf[random_idx]
        latent_idx = ldx[random_idx]
        T_points = T_points[random_idx]

        return all_points, all_sdf, latent_idx, non_empty_voxel_idx, T_points


    def re_sample(self):
        self.mesh_directory = []
        for i in range(self.n_shapes):
            surface_points = np.load(self.points_filenames[i])
            mesh = trimesh.load_mesh(self.mesh_filenames[i])
            points, sdf = sample_sdf(mesh, 500000)
            T_points, latent_idx = self.voxelize_points(points, self.n_voxels)
            T_surface_points, surface_latent_idx = self.voxelize_points(surface_points, self.n_voxels)

            self.mesh_directory.append({'off_surface': {'points': points, 'sdf': sdf, 'T_points': T_points, 'latent_idx': latent_idx},\
                    'surface': {'points': surface_points, 'sdf': 0, 'T_points': T_surface_points, 'latent_idx': surface_latent_idx},
                    'non_empty_voxel_idx': np.unique(surface_latent_idx)})

        self.len = sum(x['non_empty_voxel_idx'].shape[0] for x in self.mesh_directory)
        self.bins = np.linspace(-1, 1, self.n_voxels + 1)
        self.sidelen = self.bins[1] - self.bins[0]
        self.sample_size = sample_size
        self.n_shapes = len(self.mesh_directory)

    def __len__(self):
        return self.len * 100

    def __getitem__(self, idx):
        idx = idx % self.len

        shape_idx, non_empty_idx = self.get_shape_voxel_idx(idx)
        voxel_oned_idx = self.mesh_directory[shape_idx]['non_empty_voxel_idx'][non_empty_idx]
        voxel_idx = self.unravel_idx(voxel_oned_idx)

        # voxel_center = np.asarray([self.bins[voxel_idx[i]]
                                   # for i in range(3)]) + self.sidelen / 2
        # voxel_center = voxel_center.reshape((1, 3))

        all_points = self.mesh_directory[shape_idx]['off_surface']['points']
        T_points = self.mesh_directory[shape_idx]['off_surface']['T_points']
        all_sdf = self.mesh_directory[shape_idx]['off_surface']['sdf']
        all_latent_idx = self.mesh_directory[shape_idx]['off_surface']['latent_idx']
        all_surface_points = self.mesh_directory[shape_idx]['surface']['points']
        T_surface_points = self.mesh_directory[shape_idx]['surface']['T_points']
        all_surface_latent_idx = self.mesh_directory[shape_idx]['surface']['latent_idx']

        mask = np.where(all_latent_idx[:, 0] == voxel_oned_idx)

        # mask = np.where(np.linalg.norm(
            # all_points - voxel_center, axis=1) < (self.sidelen * 0.75))
        coords = all_points[mask]

        # T_coords = (coords - voxel_center) / (self.sidelen / 2)
        T_coords = T_points[mask]
        # temp = np.ones((T_coords.shape[0], 1))
        # latent_idx = np.hstack([temp * shape_idx, temp * voxel_oned_idx])
        latent_idx = voxel_oned_idx
        sdf = all_sdf[mask]
        random_idx = np.random.randint(
            0, coords.shape[0], self.sample_size)

        # Get surface points
        # mask = np.where(np.linalg.norm(all_surface_points -
                                       # voxel_center, axis=1) < (self.sidelen))
        # mask = np.where(all_surface_latent_idx[:, 0] == voxel_oned_idx)
        # T_surface_coords = T_surface_points[mask]
        # T_surface_coords = (surface_coords - voxel_center) / (self.sidelen / 2)
        # T_surface_coords = surface_coords

        # T_surface_coords = self.get_surface_points(shape_idx, voxel_oned_idx)

        # if T_surface_coords.shape[0] < 1:
            # return self.__getitem__((idx + 1) % self.len)
        # random_idx_2 = np.random.randint(
            # 0, T_surface_coords.shape[0], self.sample_size)

        return {'coords': T_coords[random_idx],
                'global_coords': coords[random_idx],
                'voxel_idx': latent_idx,
                'shape_idx': shape_idx,
                'sdf': sdf[random_idx],
                'img': 1}

    def get_surface_points(self, shape_idx, voxel_oned_idx):
        T_surface_points = self.mesh_directory[shape_idx][4]
        all_surface_latent_idx = self.mesh_directory[shape_idx][3]
        mask = np.where(all_surface_latent_idx[:, 0] == voxel_oned_idx)
        surface_coords = T_surface_points[mask]

        return surface_coords

    def unravel_idx(self, idx):
        z = idx % self.n_voxels
        y = (idx // self.n_voxels) % self.n_voxels
        x = idx // (self.n_voxels ** 2)

        return (x, y, z)

    def voxelize_points(self, coords, n_voxels):
        bins = np.linspace(-1, 1, n_voxels + 1)
        bin_idx = np.searchsorted(bins, coords) - 1
        bin_idx = np.clip(bin_idx, 0, n_voxels - 1)
        latent_idx = bin_idx[..., :1] * n_voxels * n_voxels +\
            bin_idx[..., 1:2] * n_voxels + bin_idx[..., 2:]
        T_coords = coords - bins[bin_idx]
        sidelen = bins[1] - bins[0]
        T_coords = T_coords / sidelen
        T_coords = T_coords - 0.5
        T_coords = 2 * T_coords
        return T_coords, latent_idx


if __name__ == '__main__':
    # points_path = 'data/dragon_2_points_2_16.npy'
    # sdf_path ='data/dragon_2_sdf_2_16.npy'
    # mesh_path ='data/dragon_2_gt.ply'
    # points_path = '/home/ishit/Downloads/threedscas_points/FullBody_Decimated_points.npy'
    # sdf_path = '/home/ishit/Downloads/threedscas_points/FullBody_Decimated_sdf.npy'
    # mesh_path ='data/dragon_2_gt.ply'
    # # mesh_path ='data/car.ply'

    # # d =  DragonDataset(16, points_path, sdf_path, sample_size=512*512, extended_field = False, n_points=-1, mode='modsiren')
    # d = DynamicDragon(16, mesh_path, sample_size=16384, extended_field = False, n_points=-1, mode='modsiren')
    # ret = d[0]
    # root = '/newfoundland2/ishit/div2k/DIV2K_train_HR/'
    # filenames = '/newfoundland2/ishit/div2k/DIV2K_train_HR/train_filenames.txt'
    # d = MultiImagePatches(root, filenames, imsize=1024, cells=64, channels=3)
    # ret = d[0]

    # d = SingleShapeDataset(n_voxels=16,\
    # points_path=points_path,\
    # sdf_path=sdf_path,\
    # sample_size=512*512,\
    # n_points=500000)
    # ret = d[0]
    # points = ret['global_coords']
    # sdf = ret['sdf']

    # np.save('test_points.npy', points)
    # np.save('test_sdf.npy', sdf)
    # exit()

    root_dir = '/home/ishit/Downloads/threedscas_points/'
    d = MultiShapeDataset(16, root_dir, sample_size=512*512, extended_field = False, n_points=500000, mode='modsiren')
    ret = d[800]
    points_2 = ret['global_coords']
    # points_2 = ret['coords']
    points = ret['img']
    # sdf = ret['sdf']

    np.save('test_points.npy', points)
    np.save('test_points_2.npy', points_2)
    # np.save('test_sdf.npy', sdf)

    # path_to_video = '/home/ishit/Documents/bikes_processed/1.npy'
    # d = SingleVideoDataset(path_to_video, channels=3,
                           # cell_size=[7, 32, 32], mode='local')

    # ret = d[0]

    # import matplotlib.pyplot as plt
    # plt.imshow(ret['img'][0])
    # plt.show()
