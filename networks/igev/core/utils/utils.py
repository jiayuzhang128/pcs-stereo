import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
from scipy import interpolate

def gamma_correction(src, fGamma):
    """Gamma correction"""

    dst = src.copy()
    if len(src.shape) > 2:
        channel = src.shape[2]
        for c in range(channel):
            dst[:,:,c] = pow(src[:,:,c]/255.0, fGamma) * 255.0
    else:
        dst[:,:] = pow(src[:,:]/255.0, fGamma) * 255.0
        
    return dst

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

class InputPadder2:
    """ Pads images such that dimensions are divisible by 32 
        make sure input image is PIL object or tensor 
    """

    def __init__(self, src_dims=(582, 429), dst_dims=None, mode='downedge', divis_by=32, fill=255, pad_mode='constant'):
        if src_dims == 2:
            self.wd = src_dims[0] if src_dims[0] > src_dims[1] else src_dims[1]
            self.ht = src_dims[1] if src_dims[0] > src_dims[1] else src_dims[0]
        else:
            self.wd = src_dims[-1] if src_dims[-1] > src_dims[-2] else src_dims[-2]
            self.ht = src_dims[-2] if src_dims[-1] > src_dims[-2] else src_dims[-1]
        self.mode = mode
        self.pad_mode = pad_mode
        self.fill = fill

        if dst_dims is None:
            self.padded_ht = ((self.ht // divis_by) + 1) * divis_by
            self.padded_wd = ((self.wd // divis_by) + 1) * divis_by
            self.pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            self.pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        else:
            assert len(dst_dims) == 2
            self.padded_wd = dst_dims[0] if dst_dims[0] > dst_dims[1] else dst_dims[1] 
            self.padded_ht = dst_dims[1] if dst_dims[0] > dst_dims[1] else dst_dims[0] 
            self.pad_wd = self.padded_wd - self.wd
            self.pad_ht = self.padded_ht - self.ht

        if self.mode == 'surround':
            self._pad = [self.pad_wd//2, self.pad_ht//2, self.pad_wd - self.pad_wd//2, self.pad_ht - self.pad_ht//2]
        elif self.mode == 'downedge':
            self._pad = [0, 0, self.pad_wd, self.pad_ht]
        else:
            self._pad = [self.pad_wd//2, 0, self.pad_wd - self.pad_wd//2, self.pad_ht]

        self.unpad_region = (self._pad[0], self._pad[1], self.padded_wd - self._pad[2], self.padded_ht- self._pad[3])

    def pad(self, x):
        """pad mode: symmetric replicate edge constant"""
        return T.Pad(padding=self._pad, fill=self.fill, padding_mode=self.pad_mode)(x)

    def unpad(self, x):
        """make sure input is PIL image or tensor"""
        if isinstance(x, Image.Image):
            wd, ht = x.size
            assert (wd == self.padded_wd) and (ht == self.padded_ht)
            unpad = x.copy().crop(self.unpad_region)
        elif isinstance(x, torch.Tensor):
            ht = x.shape[-2]
            wd = x.shape[-1]
            assert (wd == self.padded_wd) and (ht == self.padded_ht)
            unpad = x.clone()[..., self.unpad_region[1]:self.unpad_region[3], self.unpad_region[0]:self.unpad_region[2]]
        else:
            print("Please make sure input is PIL image or tensor")

        return unpad

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]

    # print("$$$55555", img.shape, coords.shape)
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1

    # print("######88888", xgrid)
    assert torch.unique(ygrid).numel() == 1 and H == 1 # This is a stereo problem

    grid = torch.cat([xgrid, ygrid], dim=-1)
    # print("###37777", grid.shape)
    img = F.grid_sample(img, grid, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return  8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)

def gauss_blur(input, N=5, std=1):
    B, D, H, W = input.shape
    x, y = torch.meshgrid(torch.arange(N).float() - N//2, torch.arange(N).float() - N//2)
    unnormalized_gaussian = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * std ** 2))
    weights = unnormalized_gaussian / unnormalized_gaussian.sum().clamp(min=1e-4)
    weights = weights.view(1,1,N,N).to(input)
    output = F.conv2d(input.reshape(B*D,1,H,W), weights, padding=N//2)
    return output.view(B, D, H, W)