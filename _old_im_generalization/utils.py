import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import torch
import torch.nn
import scipy.fftpack
import time
import numpy as np
import plyfile
import skimage.measure

def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

class Sine(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


def get_hypo_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)

def get_lipschitz_loss(ground_truth, output):

    mask_idx_2 = list(range(0, hparams['imsize']-1, self.train_dataset.cell_width))[1:]
    mask_idx_1 = list(range(-1, hparams['imsize']-1, self.train_dataset.cell_width))[1:]
    mask = torch.zeros_like(ground_truth)

    mask[:, :, mask_idx_1, :] = 1
    mask[:, :, mask_idx_2, :] = 1
    mask[:, :, :, mask_idx_1] = 1
    mask[:, :, :, mask_idx_2] = 1

    loss = torch.mean( ((ground_truth - output) * mask).pow(2) )
    return loss

def mse(image_pred, image_gt, valid_mask=None, reduction='mean'):
    value = (image_pred-image_gt)**2
    if valid_mask is not None:
        value = value[valid_mask]
    if reduction == 'mean':
        return torch.mean(value)
    return value


def psnr(image_pred, image_gt, valid_mask=None, reduction='mean'):
    return -10*torch.log10(mse(image_pred, image_gt, valid_mask, reduction))


def unblockshaped(arr, h, w):
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1, 2)
               .reshape(h, w))


def unblockshaped_tensor(arr, h, w):
    bs, n, c, nrows, ncols = arr.shape
    arr = arr.permute(0, 1, 3, 4, 2)
    arr = arr.view(bs, h//nrows, -1, nrows, ncols, c)
    arr = arr.permute(0, 1, 3, 2, 4, 5).contiguous()
    arr = arr.view(bs, h, w, c).permute(0, 3, 1, 2)
    return arr
    # return arr.view(bs, h//nrows, -1, c, nrows, ncols).permute(0, 1, 3, 2, 4).contiguous().view(bs, c, h, w)

def unblockshaped_tensor_single(arr, h, w):
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .permute(0, 2, 1, 3)
               .reshape(h, w))
def colorize(value, vmin=None, vmax=None, cmap=None):

    # normalize
    vmax = 1
    vmin = 0
    # vmax = torch.max(value)
    # vmin = torch.min(value)
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # squeeze last dim if it exists
    if len(value.shape) > 2:
        value = value.squeeze(2)
    
    value = value.detach().cpu().numpy()
    cm = plt.get_cmap('viridis')

    colored_image = cm(value)

    return torch.tensor(colored_image)

def convert_sdf_samples_to_ply(
    pytorch_3d_sdf_tensor,
    voxel_grid_origin,
    voxel_size,
    ply_filename_out,
    offset=None,
    scale=None,
):
    """
    Convert sdf samples to .ply
    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to
    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    start_time = time.time()

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()

    verts, faces, normals, values = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    try:
        verts, faces, normals, values = skimage.measure.marching_cubes_lewiner(
            numpy_3d_sdf_tensor, level=0.0, spacing=[voxel_size] * 3
        )
    except:
        pass

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    ply_data.write(ply_filename_out)

def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
def blockshaped_wc(arr, nrows, ncols):
    h, w, c = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols, c)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols, c))
