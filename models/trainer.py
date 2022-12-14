import os
import wandb

import torch
import torch.nn as nn

import models.network.net as net
from models.network.glow import Glow
from models.utils import (
    get_gradients_loss,
    save_checkpoint
)

from models.scheduler import IterLRScheduler
from models.losses.tv_loss import TVLoss


class MergeModel(nn.Module):
    def __init__(self, args):
        super(MergeModel, self).__init__()
        self.glow = Glow(3, args.n_flow, args.n_block, args.affine, conv_lu=not args.no_lu)

    def forward(self, content_images, domain_class):
        z_c = self.glow(content_images, forward=True)
        stylized_images = self.glow(z_c, forward=False, style=domain_class)

        return stylized_images


class Trainer():
    def __init__(self, args) -> None:
        self.args = args
        self.init = True
        self.device = torch.device(args.device)

        self.model = MergeModel(args)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.lr_scheduler = IterLRScheduler(self.optimizer, args.lr_steps, args.lr_mults, last_iter=args.last_iter)

        vgg = net.vgg
        vgg.load_state_dict(torch.load(args.vgg, map_location=self.device))
        self.encoder = net.Net(vgg, args.keep_ratio).to(self.device)
        self.tv_loss = TVLoss().to(self.device)

        if self.args.wandb:
            self.logger = wandb

    def train(self, batch_id, content_iter, style_iter):
        content_images = content_iter.to(self.device)
        style_images = style_iter.to(self.device)
        target_style = style_iter
        min_loss = 2**16

        domain_weight = torch.tensor(1).to(self.device)

        if self.init:
            base_code = self.encoder.cat_tensor(style_images.to(self.device))
            self.model(content_images, domain_class=base_code.to(self.device))
            self.init = False
            
            return
        
        base_code = self.encoder.cat_tensor(target_style.to(self.device))
        stylized_images = self.model(content_images, domain_class=base_code.to(self.device))
        stylized_images = torch.clamp(stylized_images, 0, 1)

        if self.args.loss == "tv_loss":
            smooth_loss = self.tv_loss(stylized_images)
        else:
            smooth_loss = get_gradients_loss(self.args, stylized_images, target_style.to(self.device))

        loss_c, loss_s = self.encoder(content_images, style_images, stylized_images, domain_weight)
        loss_c = loss_c.mean().to(self.device)
        loss_s = loss_s.mean().to(self.device)

        total_loss = self.args.content_weight * loss_c + self.args.style_weight * loss_s + smooth_loss

        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        if self.args.wandb and batch_id % self.args.log_interval == 0:
            reduce_loss = total_loss.clone()
            loss_c_ = loss_c.clone()
            loss_s_ = loss_s.clone()
            smooth_loss_ = smooth_loss.clone()
            current_lr = self.lr_scheduler.get_lr()[0]

            self.logger.log({
                "epoch": batch_id,
                "current_lr": current_lr,
                "loss_c": loss_c_.item(),
                "loss_s": loss_s_.item(),
                "smooth_loss": smooth_loss_.item(),
                "total_loss": reduce_loss.item(),
                "images": wandb.Image(torch.cat((content_images.cpu(), style_images.cpu(), stylized_images.cpu()), 0))
            })
        
        if batch_id % self.args.freq_save == 0:
            save_checkpoint({
                'step': batch_id,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(self.args.output_path, self.args.job_name, "model_save", str(batch_id) + '.ckpt'))

        if total_loss.clone() < min_loss:
            min_loss == total_loss.clone()
            save_checkpoint({
                'step': batch_id,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()
            }, os.path.join(self.args.output_path, self.args.job_name, "model_save", "best_model.ckpt"))
