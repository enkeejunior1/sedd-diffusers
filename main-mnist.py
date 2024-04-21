import wandb
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image

from ema_pytorch import EMA
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision import transforms as tfs

from sedd.model import SEDD
from sedd.scheduler import Scheduler, ScoreEntropyLoss

# load config
device = torch.device('cuda')
dtype = torch.float32

config = {
    'project' : 'sedd-mnist',
    'noise' : {
        'num_train_timesteps' : 1000,
        'type'  : 'loglinear',
        'eps'   : 1e-4,
    },
    'graph' : {
        'type'  : 'absorb',
    },
    'dataset' : {
        'tokens' : 16,
        'samples' : 256,
        'batch_size' : 256,
    },
    'model' : {
        'hidden_size'   : 32,
        'cond_dim'      : 32,
        'n_heads'       : 1,
        'n_blocks'      : 2,
        'dropout'       : 0.1,
        'scale_by_sigma' : False,
    },
    'optim' : {
        'lr' : 0,
        'epochs' : 1000,
        'grad_clip_norm' : 100,
    }
}
config = OmegaConf.create(config)


def get_dataset(config):
    # load dataset
    trans = tfs.Compose([tfs.ToTensor(), tfs.Normalize(mean=[0.0], std=[1/config.dataset.tokens])])
    ds = datasets.MNIST(root='/mnt/image-net-full/gayoung.lee/yonghyun.park/', train=True, download=True, transform=trans)
    
    # use only subset of dataset
    ds.data = ds.data[:config.dataset.samples]
    ds.targets = ds.targets[:config.dataset.samples]

    # make as datalaoder
    dl = torch.utils.data.DataLoader(ds, batch_size=config.dataset.batch_size)
    return dl


def validation(model, scheduler, config):
    scheduler.set_timesteps(num_inference_steps=1000, offset=0, device=device)

    num_batch = 5
    size = 28

    xt = (scheduler.num_vocabs - 1) * torch.ones(num_batch, size**2, dtype=torch.long) # base distribution
    xt = xt.to(device)

    for t in tqdm(scheduler.timesteps):
        if t == scheduler.timesteps[999]:
            break

        with torch.no_grad():
            # forward
            t = torch.tensor([t], device=xt.device)
            score = model(xt, t).exp()
            
            # step
            xt = scheduler.step(score, t, xt)

    # return as images
    images = [
        Image.fromarray(
            (255 / config.dataset.tokens * xt[i])
            .view(size, size).cpu().numpy().astype(np.uint8)
        )
        for i in range(num_batch)
    ]

    return images


def init_wandb(config):
    wandb.login()
    wandb.init(
        project=config.project, 
        config={
            'lr': config.optim.lr,
            'batch_size': config.dataset.batch_size,
        },
        name=f'mnist-tokens_{config.dataset.tokens}-samples_{config.dataset.samples}-lr_{config.optim.lr}-batch_size_{config.dataset.batch_size}',
    )

def main(config):
    # wand init
    init_wandb(config)

    # load dataset
    dl = get_dataset(config)
    
    # load model
    model = SEDD(config)
    ema = EMA(model, beta=0.9999, update_after_step=100, update_every=10)

    # load scheduler, loss function
    scheduler = Scheduler(config)
    loss_fn = ScoreEntropyLoss(scheduler)

    # prepare training (optim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)

    model.to(device, dtype)
    scheduler.to(device, dtype)
    ema.to(device, dtype)

    wandb.watch(model, loss_fn, log='all', log_freq=10)
    for epoch in tqdm(range(config.optim.epochs)):
        model.train()
        for x0, _ in dl:
            x0 = x0.to(device)
            x0 = x0.flatten(start_dim=1).long()
            
            # perturb x0
            t = torch.randint(1, config.noise.num_train_timesteps, (x0.size(0),), device=device)
            xt = scheduler.add_noise(x0, t)
            
            # model forward
            log_score = model(xt, t)
            
            # compute loss function 
            loss = loss_fn(log_score, t, xt, x0)

            if loss.isnan():
                raise ValueError('loss is nan')
            
            # update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad()
            
            ema.update()

            # track
            wandb.log({"loss": loss.item()})

        # track
        if epoch % 1000 == 0:
            model.eval()
            ema.eval()
            
            images = validation(ema, scheduler, config)
            wandb.log(
                {"ema-generated images": [
                    wandb.Image(image) for image in images
                ]}
            )

            images = validation(model, scheduler, config)
            wandb.log(
                {"generated images": [
                    wandb.Image(image) for image in images
                ]}
            )

    wandb.finish()

if __name__ == '__main__':
    for lr in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        config.optim.lr = lr
        main(config)
