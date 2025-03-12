
import numpy as np
import json
import io

class DataScaler:
    def __init__(self, path_pressure='./mean_std.json',path_single='./mean_std_single.json',single_level_vnames=['u10', 'v10', 'msl','u100','v100','t2m','sst','tp6h','swvl1','mtnlwrf','swh','mwd','mwp'],multi_level_vnames=['z', 'q', 'u', 'v', 't']):
        self.single_level_vnames = single_level_vnames #['u10', 'v10', 'msl','u100','v100','t2m','sst','tp6h','swvl1','mtnlwrf','swh','mwd','mwp']
        self.multi_level_vnames = multi_level_vnames #['z', 'q', 'u', 'v', 't']
        """初始化均值和标准差"""
        self.mean_std = {}
        with open(path_pressure,mode='r') as f:
            multi_level_mean_std = json.load(f)
        with open(path_single,mode='r') as f:
            single_level_mean_std = json.load(f)
        multi_level_mean_std['mean'].update(single_level_mean_std['mean'])
        multi_level_mean_std['std'].update(single_level_mean_std['std'])
        self.mean_std['mean'] = multi_level_mean_std['mean']
        self.mean_std['std'] = multi_level_mean_std['std']
        for vname in self.single_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]
        for vname in self.multi_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]
    def got_meanstd(self, variable,height=None):
        self.height = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
            500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
        if height is not None and variable not in self.single_level_vnames:
            height_index = self.height.index(height)
            return self.mean_std['mean'][variable][height_index],self.mean_std['std'][variable][height_index]
        else:
            return self.mean_std['mean'][variable],self.mean_std['std'][variable]

import torch
import math
import time
from torchvision import utils as vutils

import numpy as np

                    ### perlin noise ###
                    ####################
                    ####################
def rand_perlin_2d(shape, res, device, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    # 256,256,2
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)), dim = -1) % 1
    # 8, 8
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
    
    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)
    
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise



                    ### pink noise ###
                    ####################
                    ####################
def pink_noise_2d(batch_size, height, width, device):
    # cpu_device = 'cpu'

    if height != 720:
        # Generate white noise
        x = torch.randn(batch_size, height, width, device=device)

        # Fourier transform white noise
        ### FFT operator on cuda will generate redundancy memory occupation ###
        # x = x.to(cpu_device)
        X = torch.fft.fftn(x, dim=(1, 2))

        # get sample frequencies #
        # freqs_x = torch.fft.fftfreq(height, device=cpu_device)
        # freqs_y = torch.fft.fftfreq(width, device=cpu_device)
        freqs_x = torch.fft.fftfreq(height, device=device)
        freqs_y = torch.fft.fftfreq(width, device=device)
        freqs_xx, freqs_yy = torch.meshgrid([freqs_x, freqs_y])

        # compute the power spectral density of pink noise
        psd = 1 / (torch.abs(freqs_xx) + torch.abs(freqs_yy) + 1e-1)

        # Apply the power spectral density to the Fourier coefficients
        X *= torch.unsqueeze(psd, 0)

        # Inverse Fourier transform the modified coefficients
        y = torch.fft.ifftn(X, dim=(1, 2)).real
        
        # Normalize the output to be between -1 and 1
        bound, _ = torch.max(torch.abs(y), dim=-1, keepdim=True)
        bound, _ = torch.max(torch.abs(bound), dim=-2, keepdim=True)
        y /= bound
    else: ### for data 1440x720, generate pink noise in a split way
        interval = 500
        # Generate white noise
        with torch.no_grad():
            splits = batch_size // interval + 1
            y = []
            freqs_x = torch.fft.fftfreq(height, device=device, requires_grad=False)
            freqs_y = torch.fft.fftfreq(width, device=device, requires_grad=False)
            freqs_xx, freqs_yy = torch.meshgrid([freqs_x, freqs_y])
            psd = 1 / (torch.abs(freqs_xx) + torch.abs(freqs_yy) + 1e-1)
            for split in range(splits):
                next_step = min((split + 1)*interval, batch_size)
                cur_step = split*interval
                x = torch.randn(next_step-cur_step, height, width, device=device, requires_grad=False)
                _X = torch.fft.fftn(x, dim=(1, 2))
                _X *= torch.unsqueeze(psd, 0)
                _y = torch.fft.ifftn(_X, dim=(1, 2)).real
                del _X
                del x
                y.append(_y.clone().detach())
                del _y
            # torch.cuda.empty_cache() # torch.cuda.memory_allocated()
            y = torch.cat(y, dim=0)
            bound, _ = torch.max(torch.abs(y), dim=-1, keepdim=True)
            bound, _ = torch.max(torch.abs(bound), dim=-2, keepdim=True)
            y /= bound
            ## release cuda memory ##
            del bound
    return y



def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant, device=torch.device("cpu")
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of shape shape with the generated noise.

    Raises:
        ValueError: If shape is not a multiple of res.
    """
    # print(res)
    delta = (res[0] / shape[-2], res[1] / shape[-1])
    d = (shape[-2] // res[0], shape[-1] // res[1])
    d_remainder = (shape[-2] % res[0], shape[-1] % res[1])
    # grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
    #          .transpose(1, 2, 0) % 1
    grid_x, grid_y = torch.meshgrid(torch.arange(0,res[0],delta[0], device=device), torch.arange(0,res[1],delta[1], device=device), indexing='ij')
    grid_x = grid_x % 1
    grid_y = grid_y % 1
    # if len(shape) == 3:
    #     grid = grid.repeat(shape[0], 0).reshape(shape[0], *(grid.shape()))
    # Gradients
    if len(shape) == 3:
        angles = 2 * torch.pi * torch.rand(shape[0], res[0]+1, res[1]+1, device=device)
        # angles = 2*np.pi*np.random.rand(shape[0], res[0]+1, res[1]+1)
    else:
        angles = 2 * torch.pi * torch.rand(res[0]+1, res[1]+1, device=device)
        # angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = torch.dstack((torch.cos(angles), torch.sin(angles))).reshape(*(angles.shape), 2)
    # gradients = np.dstack((np.cos(angles), np.sin(angles))).reshape(*(angles.shape), 2)
    if tileable[0]:
        if len(shape) == 3:
            gradients[:, -1, :] = gradients[:, 0, :]
        else:
            gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        if len(shape) == 3:
            gradients[:, :, -1] = gradients[:, :, 0]
        else:
            gradients[:,-1] = gradients[:,0]


    gradients = gradients.repeat_interleave(d[0], dim=-3).repeat_interleave(d[1], dim=-2)
    # print(gradients.shape)


    if len(shape) == 2:
        gradients = gradients.unsqueeze(0)

    # if len(shape) == 3:
    g00 = gradients[:,                      :-d[0] + d_remainder[0],                     :-d[1] + d_remainder[1]]
    g10 = gradients[:, d[0] - d_remainder[0]:                      ,                     :-d[1] + d_remainder[1]]
    g01 = gradients[:,                      :-d[0] + d_remainder[0],d[1] - d_remainder[1]:                      ]
    g11 = gradients[:, d[0] - d_remainder[0]:                      ,d[1] - d_remainder[1]:                      ]

    n00 = torch.sum(torch.dstack((grid_x  , grid_y  )) * g00, -1)
    n10 = torch.sum(torch.dstack((grid_x-1, grid_y  )) * g10, -1)
    n01 = torch.sum(torch.dstack((grid_x  , grid_y-1)) * g01, -1)
    n11 = torch.sum(torch.dstack((grid_x-1, grid_y-1)) * g11, -1)
    # Interpolation
    t_x = interpolant(grid_x)
    t_y = interpolant(grid_y)
    # t = interpolant(grid)
    n0 = n00*(1-t_x) + t_x*n10
    n1 = n01*(1-t_x) + t_x*n11
    if len(shape) == 3:
        return 2**0.5*((1-t_y)*n0 + t_y*n1)
    else:
        return 2**0.5*((1-t_y)*n0 + t_y*n1)[0]



def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant, device=torch.device("cpu")
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multiple of lacunarity**(octaves-1)*res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            (lacunarity**(octaves-1)*res).
        octaves: The number of octaves in the noise. Defaults to 1.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The, interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).

    Returns:
        A numpy array of fractal noise and of shape shape generated by
        combining several octaves of perlin noise.

    Raises:
        ValueError: If shape is not a multiple of
            (lacunarity**(octaves-1)*res).
    """
    noise = torch.zeros(shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant, device = device
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise



