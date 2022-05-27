import torch
from openspeech.encoders import OpenspeechEncoder, ConformerEncoder
from torch import nn
import typing as tp

from codes import BaseClassificationModel

class Conformer_Classifier(BaseClassificationModel):

    def __init__(self, input_dims=93,  context_dims=12):
        super().__init__()
        self.encoder = ConformerEncoder(num_classes=10, input_dim=input_dims, joint_ctc_attention=False)
        classifier_in_dims = 512 + context_dims
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )


    def forward(self, x:torch.Tensor, context:tp.Optional[torch.Tensor]=None):
        x, _ = self.encoder(x, 300).transpose(1, 2)
        x = torch.cat([x.mean(1), context], 1)
        x = self.classifier(x)
        return x