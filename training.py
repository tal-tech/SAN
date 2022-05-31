import torch
from tqdm import tqdm

from utils import updata_lr, Meter, cal_score


def train(params, model, optimizer, epoch, train_loader, writer=None):

    model.train()
    device = params['device']
    loss_meter = Meter()

    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

    with tqdm(train_loader, total=len(train_loader)) as pbar:
        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(pbar):

            images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(
                device), label_masks.to(device)

            batch, time = labels.shape[:2]
            if not 'lr_decay' in params or params['lr_decay'] == 'cosine':
                updata_lr(optimizer, epoch, batch_idx, len(train_loader), params['epoches'], params['lr'])
            optimizer.zero_grad()

            probs, loss = model(images, image_masks, labels, label_masks)

            word_loss, struct_loss, parent_loss, kl_loss = loss
            loss = (word_loss + struct_loss + parent_loss + kl_loss)

            loss.backward()
            if params['gradient_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['gradient'])

            optimizer.step()

            loss_meter.add(loss.item())

            wordRate, structRate, ExpRate = cal_score(probs, labels, label_masks)

            word_right = word_right + wordRate * time
            struct_right = struct_right + structRate * time
            exp_right = exp_right + ExpRate * batch
            length = length + time
            cal_num = cal_num + batch

            if writer:
                current_step = epoch * len(train_loader) + batch_idx + 1
                writer.add_scalar('train/loss', loss.item(), current_step)
                writer.add_scalar('train/word_loss', word_loss.item(), current_step)
                writer.add_scalar('train/struct_loss', struct_loss.item(), current_step)
                writer.add_scalar('train/WordRate', wordRate, current_step)
                writer.add_scalar('train/parent_loss', parent_loss.item(), current_step)
                writer.add_scalar('train/kl_loss', kl_loss.item(), current_step)
                writer.add_scalar('train/structRate', structRate, current_step)
                writer.add_scalar('train/ExpRate', ExpRate, current_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], current_step)

            pbar.set_description(f'Epoch: {epoch+1} train loss: {loss.item():.4f} word loss: {word_loss:.4f} '
                                 f'struct loss: {struct_loss:.4f} parent loss: {parent_loss:.4f} '
                                 f'kl loss: {kl_loss:.4f} WordRate: {word_right / length:.4f} '
                                 f'structRate: {struct_right / length:.4f} ExpRate: {exp_right / cal_num:.4f}')

        if writer:
            writer.add_scalar('epoch/train_loss', loss_meter.mean, epoch+1)
            writer.add_scalar('epoch/train_WordRate', word_right / length, epoch+1)
            writer.add_scalar('epoch/train_structRate', struct_right / length, epoch + 1)
            writer.add_scalar('epoch/train_ExpRate', exp_right / cal_num, epoch + 1)
        return loss_meter.mean, word_right / length, struct_right / length, exp_right / cal_num


def eval(params, model, epoch, eval_loader, writer=None):

    model.eval()
    device = params['device']
    loss_meter = Meter()

    word_right, struct_right, exp_right, length, cal_num = 0, 0, 0, 0, 0

    with tqdm(eval_loader, total=len(eval_loader)) as pbar, torch.no_grad():

        for batch_idx, (images, image_masks, labels, label_masks) in enumerate(eval_loader):

            images, image_masks, labels, label_masks = images.to(device), image_masks.to(device), labels.to(
                device), label_masks.to(device)

            batch, time = labels.shape[:2]

            probs, loss = model(images, image_masks, labels, label_masks, is_train=False)

            word_loss, struct_loss = loss
            loss = word_loss + struct_loss
            loss_meter.add(loss.item())

            wordRate, structRate, ExpRate = cal_score(probs, labels, label_masks)

            word_right = word_right + wordRate * time
            struct_right = struct_right + structRate * time
            exp_right = exp_right + ExpRate
            length = length + time
            cal_num = cal_num + batch

            if writer:
                current_step = epoch * len(eval_loader) + batch_idx + 1
                writer.add_scalar('eval/loss', loss.item(), current_step)
                writer.add_scalar('eval/word_loss', word_loss.item(), current_step)
                writer.add_scalar('eval/struct_loss', struct_loss.item(), current_step)
                writer.add_scalar('eval/WordRate', wordRate, current_step)
                writer.add_scalar('eval/structRate', structRate, current_step)
                writer.add_scalar('eval/ExpRate', ExpRate, current_step)

            pbar.set_description(f'Epoch: {epoch + 1} eval loss: {loss.item():.4f} word loss: {word_loss:.4f} '
                                 f'struct loss: {struct_loss:.4f} WordRate: {word_right / length:.4f} '
                                 f'structRate: {struct_right / length:.4f} ExpRate: {exp_right / cal_num:.4f}')

        if writer:
            writer.add_scalar('epoch/eval_loss', loss_meter.mean, epoch + 1)
            writer.add_scalar('epoch/eval_WordRate', word_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_structRate', struct_right / length, epoch + 1)
            writer.add_scalar('epoch/eval_ExpRate', exp_right / len(eval_loader.dataset), epoch + 1)
        return loss_meter.mean, word_right / length, struct_right / length, exp_right / cal_num