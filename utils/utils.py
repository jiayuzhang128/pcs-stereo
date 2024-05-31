import re
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from scipy import interpolate
from sklearn.decomposition import PCA
import matplotlib as mpl
import matplotlib.cm as cm
import torchvision.transforms as T

def colorize_image(x, colormap='jet'):

            gray_image = x.detach().squeeze(0).cpu().numpy()

            # find value that greater than 95% of all values as max
            vmax = np.percentile(gray_image, 95)

            normalizer = mpl.colors.Normalize(vmin=gray_image.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)

            colored_image = (mapper.to_rgba(gray_image)[:, :, :3] * 255).astype(np.uint8)
            return torch.from_numpy(colored_image).permute(2, 0, 1)

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s

def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)
  
  
def read_pfm(path):
    """Read pfm file.

    Args:
        path (str): path to file

    Returns:
        tuple: (data, scale)
    """
    with open(path, "rb") as file:

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header.decode("ascii") == "PF":
            color = True
        elif header.decode("ascii") == "Pf":
            color = False
        else:
            raise Exception("Not a PFM file: " + path)

        dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
        if dim_match:
            width, height = list(map(int, dim_match.groups()))
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().decode("ascii").rstrip())
        if scale < 0:
            # little-endian
            endian = "<"
            scale = -scale
        else:
            # big-endian
            endian = ">"

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)

        return data, scale
      
def load_image(imfile, device='cuda'):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    if len(img.shape) == 2:
        img = np.tile(img[..., np.newaxis],[1, 1, 3])
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(device)

class InputPadder:
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

class IGEVInputPadder:
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

def mean_nonzero(tensor):
    """计算所有非零张量的平均值"""
    if torch.mean(tensor) != 0:
        dims = tensor.dim()
        # 获取非零元素的索引
        non_zero_indices = torch.nonzero(tensor)
        # 提取非零元素的值
        non_zero_values = tensor[[non_zero_indices[:, idx] for idx in range(dims)]]
        # 计算非零值的平均值
        average = torch.mean(non_zero_values.float())
    else:
        return torch.mean(tensor)
    return average

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

def get_device(device=None):
    if isinstance(device, torch.device):
        return device
    return torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_map_location():
    return None if torch.cuda.is_available() else lambda storage, loc: storage

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def torch2np(tensor):
    """
    Convert from torch tensor to numpy convention.
    If 4D -> [b, c, h, w] to [b, h, w, c]
    If 3D -> [c, h, w] to [h, w, c]

    :param tensor: Torch tensor
    :return: Numpy array
    """
    d = tensor.dim()
    perm = [0, 2, 3, 1] if d == 4 else \
           [1, 2, 0] if d == 3 else \
           [0, 1]
    return tensor.permute(perm).detach().cpu().numpy()

def np2torch(array, dtype=None):
    """
    Convert a numpy array to torch tensor convention.
    If 4D -> [b, h, w, c] to [b, c, h, w]
    If 3D -> [h, w, c] to [c, h, w]

    :param array: Numpy array
    :param dtype: Target tensor dtype
    :return: Torch tensor
    """
    d = array.ndim
    perm = [0, 3, 1, 2] if d == 4 else \
           [2, 0, 1] if d == 3 else \
           [0, 1]

    tensor = torch.from_numpy(array).permute(perm)
    return tensor.type(dtype) if dtype else tensor.float()

def img2torch(img, batched=False):
    """
    Convert single image to torch tensor convention.
    Image is normalized and converted to 4D: [1, 3, h, w]

    :param img: Numpy image
    :param batched: Return as 4D or 3D (default)
    :return: Torch tensor
    """
    img = torch.from_numpy(img.astype(np.float32)).permute([2, 0, 1])
    if img.max() > 1:
        img = img/img.max()
    if batched:
        img.unsqueeze_(0)
    return img

def fmap2img(fmap, pca=None):
    """Convert n-dimensional torch feature map to an image via PCA or normalization."""
    if fmap.dim() < 4:
        fmap.unsqueeze_(0)
    b, c, h, w = fmap.shape

    if pca is None and c == 3:
        return torch2np(norm_quantize(fmap, per_channel=True))

    pca_fn = pca.transform if pca else PCA(n_components=3).fit_transform

    pca_feats = reshape_as_vectors(fmap).cpu().numpy()
    out = [pca_fn(f).reshape(h, w, 3) for f in pca_feats]
    out = [(x - x.min()) / (x.max() - x.min()) for x in out]
    return np.stack(out, axis=0)

def fmap2pca(fmap):
    """Convert n-dimensional torch feature map to an image via PCA."""
    pca = PCA(n_components=3)

    if fmap.dim() < 4:
        fmap.unsqueeze_(0)
    b, c, h, w = fmap.shape

    pca_feats = reshape_as_vectors(fmap).cpu().numpy()
    out = pca.fit(pca_feats.reshape(-1, c))
    return out

def freeze_model(model):
    """Fix all model parameters and prevent training."""
    for param in model.parameters():
        param.requires_grad = False

def unfreeze_model(model):
    """Make all model parameters trainable."""
    for params in model.parameters():
        params.requires_grad = True

def norm_quantize(tensor, per_channel=False):
    """Normalize between [0, 1] and quantize to 255 image levels."""
    if per_channel:
        t_min = tensor.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0].min(0, keepdim=True)[0]
        t_max = tensor.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0].max(0, keepdim=True)[0]
        norm = (tensor - t_min) / (t_max - t_min)
    else:
        norm = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    quant = torch.round(norm*255)/255
    return quant

def upsample_like(tensor, ref_tensor):
    """Upsample tensor to match ref_tensor shape."""
    return F.interpolate(tensor, size=ref_tensor.shape[-2:], mode='bilinear', align_corners=True)

def reshape_as_vectors(tensor):
    """Reshape from (b, c, h, w) to (b, h*w, c)."""
    b, c = tensor.shape[:2]
    return tensor.reshape(b, c, -1).permute(0, 2, 1)

def reshape_as_fmap(tensor, shape):
    """Reshape from (b, h*w, c) to (b, c, h, w)."""
    b, (h, w) = tensor.shape[0], shape
    return tensor.reshape(b, h, w, -1).permute(0, 3, 1, 2)
