import os
import logging
import argparse
import wandb

from munch import Munch
from torch.utils.data import DataLoader

from models.dataset import get_dataset
from models.sampler import DistributedGivenIterationSampler
from models.trainer import Trainer

from models.utils import (
    yaml_load, 
    parse_args, 
    seed_everything
)

from tqdm.auto import tqdm


def train(args: dict):
    print("Build dataset from data source")
    train_dataset = get_dataset(args)
    train_sampler = DistributedGivenIterationSampler(
        train_dataset,
        args.max_iter,
        args.batch_size,
        world_size=1,
        rank=0,
        last_iter=args.last_iter
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=train_sampler
    )

    dset = tqdm(iter(train_loader))
    dset.set_description("Training")

    trainer = Trainer(args)
    for batch_id, (content_iter, style_iter) in enumerate(dset):
        trainer.train(batch_id, content_iter, style_iter)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="StyleFlow GAN Train model")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config file")
    parser.add_argument("--show_model", action="store_true", help="display config load from yaml config file")
    parser.add_argument("--no_cuda", action="store_true", help="use CPU instead")
    parser.add_argument("--debug", action="store_true", help="enable DEBUG mode")
    parser.add_argument("--resume", action="store_true", help="resume training model")
    parsed_args = parser.parse_args()
    
    if parsed_args.config is None:
        parsed_args.config = os.path.realpath("./configs/debug.yaml")

    params = yaml_load(parsed_args.config)
    args = parse_args(Munch(params), **vars(parsed_args))

    model_path = os.path.join(args.output_path, args.job_name, "model_save")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        print("Make output folder to save checkpoint")

    logging.getLogger().setLevel(logging.DEBUG if parsed_args.debug else logging.WARNING)
    seed_everything(args.seed)

    if args.wandb:
        if not parsed_args.resume:
            args.id = wandb.util.generate_id()
        wandb.init(config=dict(args), project=args.project, entity=args.entity,\
             resume="allow", name=f"{args.name}_{args.id}", id=args.id)
        args = Munch(wandb.config)
    
    if args.show_model:
        print("Model config:", dict(args))

    train(args)
