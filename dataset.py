import torch
import pickle as pkl
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import cv2


class HYBTr_Dataset(Dataset):

    def __init__(self, params, image_path, label_path, words, is_train=True):
        super(HYBTr_Dataset, self).__init__()
        with open(image_path, 'rb') as f:
            self.images = pkl.load(f)
        with open(label_path, 'rb') as f:
            self.labels = pkl.load(f)

        self.name_list = list(self.labels.keys())
        self.words = words
        self.max_width = params['image_width']
        self.is_train = is_train
        self.params = params

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        name = self.name_list[idx]

        image = self.images[name]

        image = torch.Tensor(image) / 255
        image = image.unsqueeze(0)

        label = self.labels[name]

        child_words = [item.split()[1] for item in label]
        child_words = self.words.encode(child_words)
        child_words = torch.LongTensor(child_words)
        child_ids = [int(item.split()[0]) for item in label]
        child_ids = torch.LongTensor(child_ids)

        parent_words = [item.split()[3] for item in label]
        parent_words = self.words.encode(parent_words)
        parent_words = torch.LongTensor(parent_words)
        parent_ids = [int(item.split()[2]) for item in label]
        parent_ids = torch.LongTensor(parent_ids)


        struct_label = [item.split()[4:] for item in label]
        struct = torch.zeros((len(struct_label), len(struct_label[0]))).long()
        for i in range(len(struct_label)):
            for j in range(len(struct_label[0])):
                struct[i][j] = struct_label[i][j] != 'None'

        label = torch.cat([child_ids.unsqueeze(1), child_words.unsqueeze(1), parent_ids.unsqueeze(1), parent_words.unsqueeze(1), struct], dim=1)

        return image, label


def get_dataset(params):

    words = Words(params['word_path'])

    params['word_num'] = len(words)
    params['struct_num'] = 7
    print(f"training data，images: {params['train_image_path']} labels: {params['train_label_path']}")
    print(f"test data，images: {params['eval_image_path']} labels: {params['eval_label_path']}")
    train_dataset = HYBTr_Dataset(params, params['train_image_path'], params['train_label_path'], words)
    eval_dataset = HYBTr_Dataset(params, params['eval_image_path'], params['eval_label_path'], words)

    train_sampler = RandomSampler(train_dataset)
    eval_sampler = RandomSampler(eval_dataset)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=train_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                              num_workers=params['workers'], collate_fn=collate_fn, pin_memory=True)

    print(f'train dataset: {len(train_dataset)} train steps: {len(train_loader)} '
          f'eval dataset: {len(eval_dataset)} eval steps: {len(eval_loader)}')

    return train_loader, eval_loader


def collate_fn(batch_images):

    max_width, max_height, max_length = 0, 0, 0
    batch, channel = len(batch_images), batch_images[0][0].shape[0]
    proper_items = []
    for item in batch_images:
        if item[0].shape[1] * max_width > 1600 * 320 or item[0].shape[2] * max_height > 1600 * 320:
            continue
        max_height = item[0].shape[1] if item[0].shape[1] > max_height else max_height
        max_width = item[0].shape[2] if item[0].shape[2] > max_width else max_width
        max_length = item[1].shape[0] if item[1].shape[0] > max_length else max_length
        proper_items.append(item)

    images, image_masks = torch.zeros((len(proper_items), channel, max_height, max_width)), torch.zeros((len(proper_items), 1, max_height, max_width))
    labels, labels_masks = torch.zeros((len(proper_items), max_length, 11)).long(), torch.zeros((len(proper_items), max_length, 2))

    for i in range(len(proper_items)):

        _, h, w = proper_items[i][0].shape
        images[i][:, :h, :w] = proper_items[i][0]
        image_masks[i][:, :h, :w] = 1

        l = proper_items[i][1].shape[0]
        labels[i][:l, :] = proper_items[i][1]
        labels_masks[i][:l, 0] = 1

        for j in range(proper_items[i][1].shape[0]):
            labels_masks[i][j][1] = proper_items[i][1][j][4:].sum() != 0

    return images, image_masks, labels, labels_masks


class Words:
    def __init__(self, words_path):
        with open(words_path) as f:
            words = f.readlines()
            print(f'{len(words)} symbols in total')

        self.words_dict = {words[i].strip(): i for i in range(len(words))}
        self.words_index_dict = {i: words[i].strip() for i in range(len(words))}

    def __len__(self):
        return len(self.words_dict)

    def encode(self, labels):
        label_index = [self.words_dict[item] for item in labels]
        return label_index

    def decode(self, label_index):
        label = ' '.join([self.words_index_dict[int(item)] for item in label_index])
        return label
