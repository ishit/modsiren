import trimesh
import pyrender
import numpy as np
points = np.load('199664.npy')
print(points.shape)
sdf = points[:, 3]
points = points[:, :3]

# sdf = np.load('dragon_2_sdf_2.npy')
#sdf = np.load('D:/Thingi10k/SDF_samples_2/100026_sdf.npy')
# np.save('dragon_2_points_all.npy', points)
# np.save('dragon_2_sdf_all.npy', sdf)
#points = np.load('mandelbulb_points_10M.npy')
#sdf = np.load('mandelbulb_sdf_10M.npy')
# points = np.load('D:/Thingi10k/sdf/100026_points.npy')
# sdf = np.load('D:/Thingi10k/sdf/100026_sdf.npy')
#points = np.load('D:/Thingi10k/new_sdf/100026_points.npy')
#sdf = np.load('D:/Thingi10k/new_sdf/100026_sdf.npy')
# mask = (np.abs(sdf) < 0.1)
# points = points[mask]
# sdf = sdf[mask]

#points = np.load('D:/Thingi10k/sdf_samples_large/100336_points.npy')
#sdf = np.load('D:/Thingi10k/sdf_samples_large/100336_sdf.npy')
colors = np.zeros(points.shape)
colors[sdf < 0, 2] = 1
colors[sdf > 0, 0] = 1
cloud = pyrender.Mesh.from_points(points, colors=colors)

# m = pyrender.Mesh.from_trimesh(mesh)
scene = pyrender.Scene()
scene.add(cloud)
#scene.add(m)
viewer = pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)
