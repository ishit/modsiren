"""Dataset and I/O classes."""
import numpy as np
import torch as th
from torch.utils.data import Dataset
import imageio


def get_mgrid(sidelen, dim=2):
    tensors = tuple(dim * [th.linspace(-1, 1, steps=sidelen)])
    mgrid = th.stack(th.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape((-1, 2))
    return mgrid


def get_mgrid_2(sidelen, dim=2):
    pixel_edges = th.linspace(-1, 1, sidelen + 1)[:-1]
    pixel_width = pixel_edges[1] - pixel_edges[0]
    pixel_centers = pixel_edges + pixel_width / 2
    tensors = tuple(dim * [pixel_centers])
    mgrid = th.stack(th.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape((-1, 2))
    return mgrid


class SingleImagePatches(Dataset):
    """Dataset that extracts fixed-size tiles from a single image.

    This class drops the last row/column if the image dimensions are not
    multiple of the cell size.

    Args:
        impath(str): path to the image.
    """

    def __init__(self, path, cell_size=64):
        super().__init__()

        self.cell_size = cell_size

        im = imageio.imread(path)

        if im.dtype != np.uint8:
            raise ValueError("Expected 8-bit input image.")

        if len(im.shape) != 3:
            raise ValueError("Expected RGB input image.")

        im = th.from_numpy(im).float() / 255.0
        im = im.permute(2, 0, 1)

        h, w = im.shape[1:]

        # Crop image to integer multiple of cell_size
        h = (h % cell_size)*cell_size
        w = (w % cell_size)*cell_size

        im = im[:, :h, :w]

        # Pixel center global coordinates
        x_coord = (th.arange(0, w).float() + 0.5) * 1.0
        y_coord = (th.arange(0, h).float() + 0.5) * 1.0
        x_coord = x_coord.unsqueeze(0).repeat(h, 1).unsqueeze(0)
        y_coord = y_coord.unsqueeze(1).repeat(1, w).unsqueeze(0)

        # Normalized local coordinates
        x_local_coord = (x_coord % cell_size) / cell_size
        y_local_coord = (y_coord % cell_size) / cell_size

        # Normalized global coordinates
        x_coord = x_coord / w
        y_coord = y_coord / h

        # Split into tiles
        tiles = th.cat([im, x_coord, y_coord, x_local_coord,
                        y_local_coord], 0)
        tiles = tiles.unfold(1, cell_size, cell_size)
        tiles = tiles.unfold(2, cell_size, cell_size)

        # Put batch dimension first
        tiles = tiles.permute(1, 2, 0, 3, 4)
        ny, nx, = tiles.shape[:2]
        tiles = tiles.contiguous().view(ny*nx, 7, cell_size, cell_size)

        self.tiles = tiles

        self._grid_dims = np.array([h // cell_size, w // cell_size])

    def __len__(self):
        return self.tiles.shape[0]

    def __getitem__(self, idx):
        im = self.tiles[idx, :3]

        # Coordinates in [-1, 1]
        global_coords = self.tiles[idx, 3:5] * 2.0 - 1.0
        local_coords = self.tiles[idx, 5:7] * 2.0 - 1.0

        return {
            'idx': idx,
            'ref': im,
            'local_coords': local_coords,
            'global_coords': global_coords,
            'grid_dims': self._grid_dims,
        }
