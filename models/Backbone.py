import torch.nn as nn
import models
import torch

class Backbone(nn.Module):
    def __init__(self, params=None):
        super(Backbone, self).__init__()

        self.params = params
        self.use_label_mask = params['use_label_mask']

        self.encoder = getattr(models, params['encoder']['net'])(params=self.params)
        self.decoder = getattr(models, params['decoder']['net'])(params=self.params)
        self.cross = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss(reduction='none')
        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet'][
            'conv1_stride']

    def forward(self, images, images_mask, labels, labels_mask, is_train=True):

        cnn_features = self.encoder(images)
        word_probs, struct_probs, words_alphas, struct_alphas, c2p_probs, c2p_alphas = self.decoder(cnn_features, labels, images_mask, labels_mask, is_train=is_train)

        word_average_loss = self.cross(word_probs.contiguous().view(-1, word_probs.shape[-1]), labels[:,:,1].view(-1))

        struct_probs = torch.sigmoid(struct_probs)
        struct_average_loss = self.bce(struct_probs, labels[:,:,4:].float())
        if labels_mask is not None:
            struct_average_loss = (struct_average_loss * labels_mask[:,:,0][:, :, None]).sum() / (labels_mask[:,:,0].sum() + 1e-10)

        if is_train:
            parent_average_loss = self.cross(c2p_probs.contiguous().view(-1, word_probs.shape[-1]), labels[:, :, 3].view(-1))
            kl_average_loss = self.cal_kl_loss(words_alphas, c2p_alphas, labels, images_mask[:, :, ::self.ratio, ::self.ratio], labels_mask)

            return (word_probs, struct_probs), (word_average_loss, struct_average_loss, parent_average_loss, kl_average_loss)

        return (word_probs, struct_probs), (word_average_loss, struct_average_loss)

    def cal_kl_loss(self, child_alphas, parent_alphas, labels, image_mask, label_mask):

        batch_size, steps, height, width = child_alphas.shape
        new_child_alphas = torch.zeros((batch_size, steps, height, width)).to(self.params['device'])
        new_child_alphas[:, 1:, :, :] = child_alphas[:,:-1,:,:].clone()
        new_child_alphas = new_child_alphas.view((batch_size*steps, height, width))
        parent_ids = labels[:,:,2] + steps * torch.arange(batch_size)[:,None].to(self.params['device'])

        new_child_alphas = new_child_alphas[parent_ids]
        new_child_alphas = new_child_alphas.view((batch_size, steps, height, width))[:, 1:, :, :]
        new_parent_alphas = parent_alphas[:,1:,:,:]

        KL_alpha = new_child_alphas * (torch.log(new_child_alphas + 1e-10) - torch.log(new_parent_alphas + 1e-10)) * image_mask
        KL_loss = (KL_alpha.sum(-1).sum(-1) * label_mask[:,:-1, 0]).sum(-1).sum(-1) / (label_mask.sum() - batch_size)

        return KL_loss

