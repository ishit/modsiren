import os
import plyfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import trimesh

from utils.extensions.mesh2sdf2_cuda import mesh2sdf
from utils.toolbox import pcl_library
from utils.poisson import *

from modsiren.utils import convert_sdf_samples_to_ply

def sdfmeshfun(point, mesh):
    out_ker = mesh2sdf.mesh2sdf_gpu(point.contiguous(),mesh)[0]
    return out_ker
    
def trimmesh(mesh_t, residual=False):
    mesh_t = mesh_t.to("cuda:0")
    valid_triangles = mesh2sdf.trimmesh_gpu(mesh_t)
    if residual:
        valid_triangles = ~valid_triangles
    mesh_t = mesh_t[valid_triangles,:,:].contiguous()
    print("[Trimmesh] {} -> {}".format(valid_triangles.size(0),mesh_t.size(0)))
    return mesh_t
    
def save_mesh(verts, faces, ply_filename_out):
    N = 256
    voxel_grid_origin = [-1, -1, -1]
    voxel_size = 2.0 / (N - 1)
    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = voxel_grid_origin[0] + verts[:, 0]
    mesh_points[:, 1] = voxel_grid_origin[1] + verts[:, 1]
    mesh_points[:, 2] = voxel_grid_origin[2] + verts[:, 2]

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
    
def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            
            # max_verts = 0
            # largest_mesh = None
            # for g in scene_or_mesh.geometry.values():
                # if g.vertices.shape[0] > max_verts:
                    # max_verts = g.vertices.shape[0]
                    # largest_mesh = g
            
            # return largest_mesh
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def meshpreprocess_bsphere(mesh_path):
    mesh = trimesh.load_mesh(mesh_path)
    mesh = as_mesh(mesh)
    main_mesh = mesh
    # mesh = mesh.vertices[mesh.faces]
    mesh = mesh.vertices

    # mesh[:,:,1] *= -1
    mesh[:,1] *= -1
    # normalize mesh
    mesh = mesh.reshape(-1,3)
    mesh_max = np.amax(mesh, axis=0)
    mesh_min = np.amin(mesh, axis=0)
    mesh_center = (mesh_max + mesh_min) / 2
    mesh = mesh - mesh_center
    # Find the max distance to origin
    max_dist = np.sqrt(np.max(np.sum(mesh**2, axis=-1)))
    mesh_scale = 1.0 / max_dist
    mesh *= mesh_scale

    # mesh = mesh.reshape(-1,3,3)
    # main_mesh.vertices[main_mesh.faces] = mesh
    main_mesh.vertices = mesh
    # main_mesh.export('test.stl')
    return main_mesh
    # exit()
    # mesh_t = torch.from_numpy(mesh.astype(np.float32)).contiguous()
    # return mesh_t

def normalize(x):
    x /= torch.sqrt(torch.sum(x**2))
    return x

def main(args):
    device = torch.device('cuda:0')

    data_path = args.mesh_npy_path
    root= '/newfoundland2/ishit/shapenetv2/ShapeNetCore.v2/02958343/'
    # csv_file = '/newfoundland2/ishit/shapenetv2/ShapeNetCore.v2/train_filenames.txt'
    csv_file = '/newfoundland2/ishit/dualsdf_car/filenames.txt'

    
    data_list = []
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            obj_path = os.path.join(root, line.strip().split('/')[-1].split('.')[0], 'models/model_normalized.obj')
            data_list.append(obj_path)
            
    # data_list.sort()
    print(len(data_list))
    num_shapes = len(data_list)

    target_path = args.output_path
    for shape_id in range(args.resume, len(data_list)):
        print('Processing {} - '.format(shape_id), end='')
        mesh_path = data_list[shape_id]
        mesh_path_split = mesh_path.split('/')
        # shapeid = mesh_path_split[-1].split('.')[0]
        shapeid = mesh_path_split[-3]
        print(mesh_path)
        save_path = os.path.join(target_path,'{}.npy'.format(shapeid))
        if os.path.exists(save_path):
            continue
        
        start = time.time()
        
        mesh = meshpreprocess_bsphere(mesh_path)
        mesh.export(f'{target_path}/{shapeid}.stl')

def sample_in_the_grid(surface_points, n_voxels, sample_count):
    disk = False                # this parameter defines if we look for Poisson-like distribution on a disk/sphere (center at 0, radius 1) or in a square/box (0-1 on x and y)
    repeatPattern = False        # this parameter defines if we look for "repeating" pattern so if we should maximize distances also with pattern repetitions
    num_iterations = 4          # number of iterations in which we take average minimum squared distances between points and try to maximize them
    first_point_zero = disk     # should be first point zero (useful if we already have such sample) or random
    iterations_per_point = 64  # iterations per point trying to look for a new point with larger distance
    sorting_buckets = 0         # if this option is > 0, then sequence will be optimized for tiled cache locality in n x n tiles (x followed by y)
    num_dim = 3                 # 1, 2, 3 dimensional version
    num_rotations = 1           # number of rotations of pattern to check against
    poisson_generator = PoissonGenerator(num_dim, disk, repeatPattern, first_point_zero)

    bins = np.linspace(-1, 1, n_voxels + 1)
    bin_idx = np.searchsorted(bins, surface_points) - 1
    bin_idx = np.clip(bin_idx, 0, n_voxels - 1)

    occupied_idx = np.unique(bin_idx, axis = 0)
    samples_per_voxel = sample_count // occupied_idx.shape[0]
    num_points = samples_per_voxel              # number of points we are looking for

    all_points = []
    sidelen = bins[1] - bins[0]
    for i in range(occupied_idx.shape[0]):
        #curr_points = sidelen * poisson_generator.find_point_set(num_points, num_iterations, iterations_per_point, num_rotations)
        curr_points = sidelen * np.random.uniform(0, 1, size=(samples_per_voxel, 3))
        T_curr_points = curr_points + bins[occupied_idx[i:i+1]]
        all_points.append(T_curr_points)
    
    return np.concatenate(all_points)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample SDF values from meshes. All the NPY files under mesh_npy_path and its child dirs will be converted and the directory structure will be preserved.')
    parser.add_argument('mesh_npy_path', type=str,
                        help='The dir containing meshes in NPY format [ #triangles x 3(vertices) x 3(xyz) ]')
    parser.add_argument('output_path', type=str,
                        help='The output dir containing sampled SDF in NPY format [ #points x 4(xyzd) ]')
    parser.add_argument('--notrim', default=False, action='store_true')
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args()
    main(args)

