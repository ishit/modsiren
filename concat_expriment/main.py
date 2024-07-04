import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from scipy import fftpack
import radialProfile
import brewer2mpl

W_1_2 = np.random.randn(2, 2)  * 4
W_1 = np.random.randn(4, 2) * 4
W_2 = np.random.randn(2, 1) * 4
W_3 = np.random.randn(2, 2) * 4
b_1 = np.random.randn(1, 2) 

xx, yy = np.meshgrid(np.linspace(-1, 1, 250), np.linspace(-1, 1, 250))
x = np.stack([xx, yy]).transpose((1, 2, 0)).reshape((-1, 2))

z_1 = np.linspace(-2, 2, 10)
z_2 = np.linspace(-2, 2, 10)

def psd(img):

    n = int( math.ceil(img.shape[0] / 2.) * 2 )

    a = np.fft.rfft(img,n, axis=0)

    a = a.real*a.real + a.imag*a.imag
    a = a.sum(axis=1)/a.shape[1]

    f = np.fft.rfftfreq(n)

    freqs = f[1:]
    amps = a[1:]
    n = int( math.ceil(img.shape[1] / 2.) * 2 )

    a = np.fft.rfft(img,n,axis=1)

    a = a.real*a.real + a.imag*a.imag
    a = a.sum(axis=0)/a.shape[0]

    f = np.fft.rfftfreq(n)
    freqs = np.concatenate([freqs, f[1:]])
    amps = np.concatenate([amps, a[1:]])
    return freqs, amps

def psd2(img):
    F1 = fftpack.fft2(img)
    F2 = fftpack.fftshift(F1)

    psd2D = np.abs(F2) ** 2
    psd1D = radialProfile.azimuthalAverage(psd2D)
    
    plt.imshow(psd2D * 10)
    plt.show()

cntr = 0

x_f = np.asarray([0])
y_f = np.asarray([0])

bmap = brewer2mpl.get_map('YlGnBu', 'Sequential', 7)
for i in range(z_1.shape[0]):
    for j in range(z_2.shape[0]):
        attn_z = np.array([[z_1[i], z_2[j]]]).dot(W_3) 
        x_concat = np.hstack([x, np.ones((x.shape[0], 1)) * z_1[i], np.ones((x.shape[0], 1)) * z_2[j]])
        o_concat = np.sin(x_concat.dot(W_1) + b_1).dot(W_2)
        o_attn = np.sin(x.dot(W_1_2) * attn_z + b_1).dot(W_2)

        f_map_attn = np.sin(o_attn).reshape((250, 250))
        f_map_concat = np.sin(o_concat).reshape((250, 250))
        plt.imsave(f'attn_{cntr:02d}.png', f_map_attn, cmap=bmap.mpl_colormap)
        plt.imsave(f'concat_{cntr:02d}.png', f_map_concat, cmap=bmap.mpl_colormap)

        cntr += 1
