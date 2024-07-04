import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import trimesh

from extensions.mesh2sdf2_cuda import mesh2sdf
from toolbox import pcl_library
from poisson import *

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
    mesh = mesh.vertices[mesh.faces]

    mesh[:,:,1] *= -1
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
    mesh = mesh.reshape(-1,3,3)
    mesh_t = torch.from_numpy(mesh.astype(np.float32)).contiguous()
    return mesh_t

def normalize(x):
    x /= torch.sqrt(torch.sum(x**2))
    return x

def main(args):
    device = torch.device('cuda:0')

    data_path = args.mesh_npy_path
    root= '/newfoundland2/ishit/shapenetv2/ShapeNetCore.v2/'
    csv_file = '/newfoundland2/ishit/shapenetv2/ShapeNetCore.v2/train_filenames.txt'
    
    data_list = []
    # with os.scandir(data_path) as npy_list:
        # for npy_path in npy_list:
            # if npy_path.is_file():
                # data_list.append(npy_path.path)

    # with open(csv_file, 'r') as f:
        # lines = f.readlines()
        # for line in lines:
            # obj_path = os.path.join(root, line.strip())
            # data_list.append(obj_path)
            
    data_list.sort()
    print(len(data_list))
    num_shapes = len(data_list)

    target_path = args.output_path
    
    # for each mesh, sample points within bounding sphere.
    # According to DeepSDF paper, 250,000x2 points around the surface,
    # 25,000 points within the unit sphere uniformly
    # To sample points around the surface, 
    #   - sample points uniformly on the surface,
    #   - Perturb the points with gaussian noise var=0.0025 and 0.00025
    #   - Then compute SDF
    num_surface_samples = 250000
    num_sphere_samples = 25000
    target_samples = 250000

    noise_vec = torch.empty([num_surface_samples,3], dtype=torch.float32, device=device) # x y z
    noise_vec2 = torch.empty([num_sphere_samples,3], dtype=torch.float32, device=device) # x y z
    noise_vec3 = torch.empty([num_sphere_samples,1], dtype=torch.float32, device=device) # x y z

    # for shape_id in range(args.resume, len(data_list)):
    for i in range(10):
        # print('Processing {} - '.format(shape_id), end='')
        # mesh_path = data_list[shape_id]
        mesh_path = '/home/ishit/Documents/localImplicit_release/data/car.ply'
        # mesh_path_split = mesh_path.split('/')
        # shapeid = mesh_path_split[-1].split('.')[0]
        # shapeid = mesh_path_split[-3]
        shapeid = '12'
        print(mesh_path)
        # save_path = os.path.join(target_path,'{}.npy'.format(shapeid))
        # if os.path.exists(save_path):
            # continue
        start = time.time()
        
        mesh = meshpreprocess_bsphere(mesh_path).to(device)
        if not args.notrim:
            # Remove inside triangles
            mesh = trimmesh(mesh)
        pcl = torch.from_numpy(pcl_library.mesh2pcl(mesh.cpu().numpy(), num_surface_samples)).to(device) # [N, 3]
        
        # Surface points
        noise_vec.normal_(0, np.sqrt(0.005))
        points1 = pcl + noise_vec
        noise_vec.normal_(0, np.sqrt(0.0005))
        points2 = pcl + noise_vec
        
        # Unit sphere points
        # for global methods
        noise_vec2.normal_(0, 1)
        shell_points = noise_vec2 / torch.sqrt(torch.sum(noise_vec2**2, dim=-1, keepdim=True))
        noise_vec3.uniform_(0, 1) # r = 1
        points3 = shell_points * (noise_vec3**(1/3))
        points3 = points3.to(device)

        # for local methods
        # points3 = sample_in_the_grid(pcl.detach().cpu().numpy(), n_voxels, num_sphere_samples)
        # points3 = torch.from_numpy(points3).float().to('cuda:0')

        all_points = torch.cat([points1, points2, points3], dim=0)
        sample_dist = sdfmeshfun(all_points, mesh)
        
        print(sample_dist)
        exit()
        
        xyzd = torch.cat([all_points, sample_dist.unsqueeze(-1)], dim=-1).cpu().numpy()
        
        xyzd_sur = xyzd[:num_surface_samples*2]
        xyzd_sph = xyzd[num_surface_samples*2:]
        
        inside_mask = (xyzd_sur[:,3] <= 0)
        outside_mask = np.logical_not(inside_mask)

        inside_cnt = np.count_nonzero(inside_mask)
        outside_cnt = np.count_nonzero(outside_mask)
        inside_stor = [xyzd_sur[inside_mask,:]]
        outside_stor = [xyzd_sur[outside_mask,:]]
        n_attempts = 0
        badsample = False
        while (inside_cnt < target_samples) or (outside_cnt < target_samples):
            noise_vec.normal_(0, np.sqrt(0.005))
            points1 = pcl + noise_vec
            noise_vec.normal_(0, np.sqrt(0.0005))
            points2 = pcl + noise_vec
            all_points = torch.cat([points1, points2], dim=0)
            sample_dist = sdfmeshfun(all_points, mesh)
            xyzd_sur = torch.cat([all_points, sample_dist.unsqueeze(-1)], dim=-1).cpu().numpy()
            inside_mask = (xyzd_sur[:,3] <= 0)
            outside_mask = np.logical_not(inside_mask)
            inside_cnt += np.count_nonzero(inside_mask)
            outside_cnt += np.count_nonzero(outside_mask)
            inside_stor.append(xyzd_sur[inside_mask,:])
            outside_stor.append(xyzd_sur[outside_mask,:])
            n_attempts += 1
            print(" - {}nd Attempt: {} / {}".format(n_attempts, inside_cnt, target_samples))
            if n_attempts > 200 or ((np.minimum(inside_cnt, outside_cnt)/n_attempts) < 500):
                with open('bads_list_{}.txt'.format(classid), 'a+') as f:
                    f.write('{},{},{},{}\n'.format(classid, shapeid, np.minimum(inside_cnt, outside_cnt), n_attempts))
                badsample = True
                break
            
        xyzd_inside = np.concatenate(inside_stor, axis=0)
        xyzd_outside = np.concatenate(outside_stor, axis=0)
        
        # num_yields = np.minimum(xyzd_inside.shape[0], xyzd_outside.shape[0])
        # num_yields = np.minimum(num_yields, target_samples // 2)
        # xyzd_inside = xyzd_inside[:num_yields,:]
        # xyzd_outside = xyzd_outside[:num_yields,:]
        
        # xyzd = np.concatenate([xyzd_inside, xyzd_outside], axis=0)
        num_yields = np.minimum(xyzd_inside.shape[0], xyzd_outside.shape[0])
        xyzd_inside = xyzd_inside[:num_yields,:]
        xyzd_outside = xyzd_outside[:num_yields,:]
        xyzd = np.concatenate([xyzd_inside, xyzd_outside], axis=0)

        xyzd_all = np.concatenate([xyzd, xyzd_sph], axis=0)
        
        end = time.time()
        print("[Perf] time: {}, yield: {}".format(end - start, num_yields))

        if not os.path.exists(target_path):
            os.makedirs(target_path)
        np.save(os.path.join(target_path,'{}.npy'.format(shapeid)), xyzd_all)

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
