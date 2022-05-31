import torch.nn as nn
import models
from infer.san_decoder import SAN_decoder

class Backbone(nn.Module):
    def __init__(self, params=None):
        super(Backbone, self).__init__()

        self.params = params
        self.use_label_mask = params['use_label_mask']

        self.encoder = getattr(models, params['encoder']['net'])(params=self.params)
        self.decoder = SAN_decoder(params=self.params)
        self.ratio = params['densenet']['ratio'] if params['encoder']['net'] == 'DenseNet' else 16 * params['resnet'][
            'conv1_stride']

    def forward(self, images, images_mask):

        cnn_features = self.encoder(images)
        prediction = self.decoder(cnn_features, images_mask)

        return prediction


