import os
import yaml
import random
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms as T

from munch import Munch
from skimage.metrics import structural_similarity as ssim


def yaml_load(file_config: str):
    with open(file_config, 'r', encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def read_text(path_file):
    with open(path_file, 'r', encoding="utf-8") as f:
        data = f.readlines()
        return [text.strip("\n") for text in data]


def parse_args(args, **kwargs) -> Munch:
    args = Munch(args)
    kwargs = Munch(**kwargs)
    args.update(kwargs)

    args.wandb = not kwargs.debug and not args.debug
    args.device = get_device(args, kwargs.no_cuda)

    return args


def get_device(args, no_cuda=False):
    device = "cpu"
    available_gpus = torch.cuda.device_count()
    args.gpu_devices = args.gpu_devices if args.get("gpu_devices", False) else list(range(available_gpus))

    if available_gpus > 0 and not no_cuda:
        device = "cuda:%d" % args.gpu_devices[0] if args.gpu_devices else 0
        assert available_gpus >= len(args.gpu_devices), "Available %d gpu, but specified gpu %s." % (available_gpus, ','.join(map(str, args.gpu_devices)))
        assert max(args.gpu_devices) < available_gpus, "legal gpu_devices should in [%s], received [%s]" % (','.join(map(str, range(available_gpus))), ','.join(map(str, args.gpu_devices)))
    
    return device


def seed_everything(seed: int):
    """Seed all RNGs
    Args:
        seed (int): seed
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def get_smooth(args, I, direction):
    weights = torch.tensor(
        [[0., 0.],
        [-1., 1.]]
    ).to(torch.device(args.device))

    weights_x = weights.view(1, 1, 2, 2).repeat(1, 1, 1, 1)
    weights_y = torch.transpose(weights_x, 0, 1)

    if direction == 'x':
        weights = weights_x
    elif direction == 'y':
        weights = weights_y

    output = torch.abs(F.conv2d(I, weights, stride=1, padding=1))
    return output


def avg(args, R, direction):
    return nn.AvgPool2d(kernel_size=3, stride=1, padding=1)(get_smooth(args, R, direction))


def get_gradients_loss(args, I, R):
    R_gray = torch.mean(R, dim=1, keepdim=True)
    I_gray = torch.mean(I, dim=1, keepdim=True)
    gradients_I_x = get_smooth(args, I_gray, 'x')
    gradients_I_y = get_smooth(args, I_gray, 'y')

    return torch.mean(gradients_I_x * torch.exp(-10 * avg(args, R_gray, 'x')) + \
        gradients_I_y * torch.exp(-10 * avg(args, R_gray, 'y')))


def save_checkpoint(state, filename):
    torch.save(state, filename)


def remove_prefix(state_dict, prefix):
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def get_checkpoint(model_path, device, prefix: str = None):
    checkpoint = torch.load(model_path, map_location=device)
    if prefix:
        checkpoint["state_dict"] = remove_prefix(checkpoint["state_dict"], prefix)

    return checkpoint


def tensor2array(tensor: torch.tensor):
    transform = T.ToPILImage()
    tensor = tensor[0, 0, ...]
    array = np.array(transform(tensor))

    return array


def get_ssim(imgA, imgB):
    imgA = tensor2array(imgA)
    imgB = tensor2array(imgB)

    return ssim(imgA, imgB)
