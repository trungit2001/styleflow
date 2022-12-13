import os
import logging
import argparse
import wandb
import torch
import numpy as np

from munch import Munch
from torch.utils.data import DataLoader

from models.dataset import get_dataset
from models.sampler import DistributedGivenIterationSampler
from models.network import net
from models.trainer import MergeModel

from models.utils import (
    yaml_load, 
    parse_args, 
    seed_everything,
    get_checkpoint,
    get_ssim
)

from tqdm.auto import tqdm


def test(args: dict):
    print("Build dataset from data source")
    test_dataset = get_dataset(args)
    test_sampler = DistributedGivenIterationSampler(
        test_dataset,
        args.max_iter,
        args.batch_size,
        world_size=1,
        rank=0,
        last_iter=args.last_iter
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=test_sampler
    )

    model = MergeModel(args)
    if os.path.isfile(args.checkpoint):
        print("Loading checkpoint")
        checkpoint = get_checkpoint(args.checkpoint, args.device, prefix="module.")
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded checkpoint found in", args.checkpoint)
    else:
        raise("No checkpoint found", args.checkpoint)

    vgg = net.vgg
    vgg.load_state_dict(get_checkpoint(args.vgg, args.device))
    encoder = net.Net(vgg).to(args.device)

    model.to(args.device)
    model.eval()

    dset = tqdm(iter(test_loader))
    desc_verbose = "Iter: {}, ssim: {:.3f}, avg_ssim: {:.3f}"
    dset.set_description("Testing")

    total_ssim = []
    for batch_id, (content_iter, style_iter) in enumerate(dset):
        content_images = content_iter.to(args.device)
        style_images = style_iter.to(args.device)

        base_code = encoder.cat_tensor(style_images)
        stylized_images = model(content_images, domain_class=base_code.to(args.device))
        stylized_images = torch.clamp(stylized_images, 0, 1)

        ssim = get_ssim(content_images, stylized_images)
        total_ssim.append(ssim)
        avg_ssim = np.mean(total_ssim)

        if args.wandb and (batch_id + 1) % args.log_interval == 0:
            wandb.log({
                "epoch": batch_id,
                "ssim": avg_ssim,
                "images": wandb.Image(torch.cat((
                    content_images.cpu(), 
                    style_images.cpu(), 
                    stylized_images.cpu()
                ), 0))
            })
        dset.set_description(desc_verbose.format(batch_id, ssim, avg_ssim))
        
    print("Structural Similarity Index Measure:", avg_ssim)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleFlow GAN Test model")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--show_model", action="store_true", help="display config load from yaml config file")
    parser.add_argument("--no_cuda", action="store_true", help="use CPU instead")
    parser.add_argument("--debug", action="store_true", help="enable DEBUG mode")
    parser.add_argument("--resume", action="store_true", help="resume training model")
    parsed_args = parser.parse_args()
    
    if parsed_args.config is None:
        parsed_args.config = os.path.realpath("./configs/debug.yaml")

    params = yaml_load(parsed_args.config)
    args = parse_args(Munch(params), **vars(parsed_args))

    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)

    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), project=args.project, entity=args.entity,\
             resume="allow", name=f"{args.name}_test_{args.id}", id=args.id)
        args = Munch(wandb.config)
    
    if args.show_model:
        print("Model config:", dict(args))

    test(args)
