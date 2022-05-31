import os
import yaml
import math
import torch
import numpy as np
from difflib import SequenceMatcher


def load_config(yaml_path):
    try:
        with open(yaml_path, 'r') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
    except:
        print('try UTF-8 encoding')
        with open(yaml_path, 'r', encoding='UTF-8') as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    if not params['experiment']:
        print('expriment name cannot be empty!')
        exit(-1)

    if not params['train_image_path']:
        print('training images cannot be empty!')
        exit(-1)

    if not params['train_label_path']:
        print('training labels cannot be empty!')
        exit(-1)

    if not params['eval_image_path']:
        print('test images cannot be empty!')
        exit(-1)

    if not params['eval_label_path']:
        print('test labels cannot be empty!')
        exit(-1)

    if not params['word_path']:
        print('word dict cannot be empty')
        exit(-1)
    return params


def updata_lr(optimizer, current_epoch, current_step, steps, epoches, initial_lr):
    if current_epoch < 1:
        new_lr = initial_lr / steps * (current_step + 1)

    else:
        new_lr = 0.5 * (1 + math.cos((current_step + 1 + (current_epoch - 1) * steps) * math.pi / (epoches * steps))) * initial_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def save_checkpoint(model, optimizer, word_score, struct_score, ExpRate_score, epoch, optimizer_save=False, path='checkpoints', multi_gpu=False, local_rank=0):

    filename = f'{os.path.join(path, model.name)}/{model.name}_WordRate-{word_score:.4f}_structRate-{struct_score:.4f}_ExpRate-{ExpRate_score:.4f}_{epoch}.pth'

    if optimizer_save:
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    else:
        state = {
            'model': model.state_dict()
        }

    torch.save(state, filename)
    print(f'Save checkpoint: {filename}\n')
    return filename


def load_checkpoint(model, optimizer, path):

    state = torch.load(path, map_location='cpu')

    if 'optimizer' in state:
        optimizer.load_state_dict(state['optimizer'])
    else:
        print(f'No optimizer in the pretrained model')

    model.load_state_dict(state['model'])


class Meter:
    def __init__(self, alpha=0.9):
        self.nums = []
        self.exp_mean = 0
        self.alpha = alpha

    @property
    def mean(self):
        return np.mean(self.nums)

    def add(self, num):
        if len(self.nums) == 0:
            self.exp_mean = num
        self.nums.append(num)
        self.exp_mean = self.alpha * self.exp_mean + (1 - self.alpha) * num

def cal_score(probs, labels, mask):

    batch_size = probs[0].shape[0]
    word_probs, struct_probs = probs
    word_label, struct_label = labels[:,:,1], labels[:,:,4:]
    struct_label = struct_label.contiguous().view(batch_size, -1)
    line_right = 0
    _, word_pred = word_probs.max(2)

    struct_mask = mask[:,:,1]
    struct_probs = struct_probs * struct_mask[:,:,None]
    struct_probs = struct_probs.contiguous().view(batch_size, -1)
    struct_pred = struct_probs > 0.5

    word_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
              for s1, s2, s3 in zip(word_label.cpu().detach().numpy(), word_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]
    struct_scores = [SequenceMatcher(None, s1[:int(np.sum(s3))], s2[:int(np.sum(s3))], autojunk=False).ratio() * (len(s1[:int(np.sum(s3))]) + len(s2[:int(np.sum(s3))])) / len(s1[:int(np.sum(s3))]) / 2
                   for s1, s2, s3 in zip(struct_label.cpu().detach().numpy(), struct_pred.cpu().detach().numpy(), mask.cpu().detach().numpy())]

    batch_size = len(word_scores) if word_probs is not None else len(struct_scores)

    for i in range(batch_size):
        if struct_mask[i].sum() > 0:
            if word_scores[i] == 1 and struct_scores[i] == 1:
                line_right += 1
        else:
            if word_scores[i] == 1:
                line_right += 1

    ExpRate = line_right / batch_size

    word_scores = np.mean(word_scores) if word_probs is not None else 0
    struct_scores = np.mean(struct_scores) if struct_probs is not None else 0
    return word_scores, struct_scores, ExpRate

