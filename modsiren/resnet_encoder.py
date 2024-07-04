import torch
from torch import nn
import torch.nn.functional as F
import functools

class ResnetGenerator(nn.Module):
    def __init__(
            self, input_nc, patch_res, latent_dim, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
            n_blocks=6, gpu_ids=[], use_parallel=True, learn_residual=False, padding_type='reflect'):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = input_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_parallel = use_parallel
        self.learn_residual = learn_residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        model += [
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(256),
            nn.ReLU(True)
        ]

        for i in range(n_blocks):
            model += [
                ResnetBlock(256, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
            ]

        self.linear = nn.Linear(256 * ((patch_res // 4) ** 2), latent_dim)
        self.model = nn.Sequential(*model)

    def forward(self, x):
        o = self.model(x)
        o = self.linear(o.view(o.shape[0], -1))
        return o

class ResnetBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()

        padAndConv = {
                'reflect': [
        nn.ReflectionPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
                'replicate': [
        nn.ReplicationPad2d(1),
        nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias)],
                'zero': [
        nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias)]
        }

        blocks = padAndConv[padding_type] + [norm_layer(dim), nn.ReLU(True)] + [nn.Dropout(0.5)] 
        self.conv_block = nn.Sequential(*blocks)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

if __name__ == '__main__':
    encoder = ResnetGenerator(3, 3, 32, 128)
    a = torch.randn((64, 3, 32, 32))

    b = encoder(a)
    print(b.shape)
