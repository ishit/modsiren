import os
import glob
import numpy as np
import trimesh
from argparse import ArgumentParser 
from modsiren.utils import compute_trimesh_chamfer

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gt_dir')
    parser.add_argument('--results')
    # parser.add_argument('--est_dir')
    args = parser.parse_args()
    root_dir = '/newfoundland2/ishit/bak_logs_chair_4/test/'
    ffn_s1_dir = 'ffn_car_s1'
    ffn_s10_dir = 'ffn_car'
    relu_dir = 'localrelu_car_all'
    ms_dir = 'modsiren_car_concat_all'

    count = 0
    ffn_score = []
    ms_score = []


    for f in os.listdir(args.gt_dir):
        shapeid = f.split('.')[-2]
        gt_filename = os.path.join(args.gt_dir, f)


        # ffn_s1_filename = os.path.join(root_dir, ffn_s1_dir, shapeid + '.ply')
        ms_filename = os.path.join(root_dir, ms_dir, shapeid + '.ply')
        ffn_s10_filename = os.path.join(root_dir, ffn_s10_dir, shapeid + '.ply')
        relu_filename = os.path.join(root_dir, relu_dir, shapeid + '.ply')
        # ms_filename = os.path.join(args.est_dir, shapeid + '.ply')
        if not os.path.exists(ffn_s10_filename) or not os.path.exists(ms_filename) or not os.path.exists(ms_filename):
            continue

        # print(ffn_s10_filename)
        # print(ms_filename)
        # print(ms_filename)
        gt_mesh = trimesh.load_mesh(gt_filename)
        ffn_mesh = trimesh.load_mesh(ffn_s10_filename)
        ms_mesh = trimesh.load_mesh(ms_filename)
        relu_mesh = trimesh.load_mesh(relu_filename)

        ffn_chamfer = compute_trimesh_chamfer(gt_mesh, ffn_mesh, 30000)
        ms_chamfer = compute_trimesh_chamfer(gt_mesh, ms_mesh, 30000)
        relu_chamfer = compute_trimesh_chamfer(gt_mesh, relu_mesh, 30000)


        result_line = f'{shapeid}, {ffn_chamfer:.10f}, {ms_chamfer:.10f}, {relu_chamfer:.10f}\n'
        print(result_line)

        with open(args.results, 'a') as f:
            f.write(result_line)

    print('Median', np.median(ffn_score))
    print('Mean', np.mean(ffn_score))
    print('Std', np.std(ffn_score))
    print('Median', np.median(ms_score))
    print('Mean', np.mean(ms_score))
    print('Std', np.std(ms_score))
