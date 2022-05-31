import os
import time
import argparse
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

from utils import load_config, save_checkpoint, load_checkpoint
from dataset import get_dataset
from models.Backbone import Backbone
from training import train, eval

parser = argparse.ArgumentParser(description='HYB Tree')
parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
parser.add_argument('--check', action='store_true', help='only for code check')
args = parser.parse_args()

if not args.config:
    print('please provide config yaml')
    exit(-1)

"""config"""
params = load_config(args.config)

"""random seed"""
random.seed(params['seed'])
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])
torch.cuda.manual_seed(params['seed'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
params['device'] = device

train_loader, eval_loader = get_dataset(params)

model = Backbone(params)
now = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
model.name = f'{params["experiment"]}_{now}_Encoder-{params["encoder"]["net"]}_Decoder-{params["decoder"]["net"]}_' \
             f'max_size-{params["image_height"]}-{params["image_width"]}'
print(model.name)
model = model.to(device)

if args.check:
    writer = None
else:
    writer = SummaryWriter(f'{params["log_dir"]}/{model.name}')

optimizer = getattr(torch.optim, params['optimizer'])(model.parameters(), lr=float(params['lr']),
                                                      eps=float(params['eps']), weight_decay=float(params['weight_decay']))

if params['finetune']:

    print('loading pretrain model weight')
    print(f'pretrain model: {params["checkpoint"]}')
    load_checkpoint(model, optimizer, params['checkpoint'])

if not args.check:
    if not os.path.exists(os.path.join(params['checkpoint_dir'], model.name)):
        os.makedirs(os.path.join(params['checkpoint_dir'], model.name), exist_ok=True)
    os.system(f'cp {args.config} {os.path.join(params["checkpoint_dir"], model.name, model.name)}.yaml')


min_score = 0
min_step = 0
for epoch in range(params['epoches']):

    train_loss, train_word_score, train_node_score, train_expRate = train(params, model, optimizer, epoch, train_loader, writer=writer)
    if epoch > 150:
        eval_loss, eval_word_score, eval_node_score, eval_expRate = eval(params, model, epoch, eval_loader, writer=writer)

        print(f'Epoch: {epoch+1}  loss: {eval_loss:.4f}  word score: {eval_word_score:.4f}  struct score: {eval_node_score:.4f} '
              f'ExpRate: {eval_expRate:.4f}')

        if eval_expRate > min_score and not args.check:
            min_score = eval_expRate
            save_checkpoint(model, optimizer, eval_word_score, eval_node_score, eval_expRate, epoch+1,
                            optimizer_save=params['optimizer_save'], path=params['checkpoint_dir'])
            min_step = 0

        elif min_score != 0 and 'lr_decay' in params and params['lr_decay'] == 'step':

            min_step += 1

            if min_step > params['step_ratio']:
                new_lr = optimizer.param_groups[0]['lr'] / params['step_decay']

                if new_lr < params['lr'] / 1000:
                    print('lr is too small')
                    exit(-1)

                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

                min_step = 0











