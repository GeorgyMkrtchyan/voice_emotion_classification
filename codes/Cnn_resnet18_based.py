import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from codes import BaseClassificationModel


class CNNClassifier_resnet_based(BaseClassificationModel):
    def __init__(self):
        super(CNNClassifier_resnet_based, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18.train(False)
        self.linear = nn.Linear(512, 10)
        self.pool = nn.AvgPool2d(8, stride=1)


    def forward(self, x):
        x = self.resnet18.conv1(x.unsqueeze(1).repeat(1,3,1,1).float())
        x = self.resnet18.bn1(x)
        x = self.resnet18.relu(x)
        x = self.resnet18.maxpool(x)

        c2 = self.resnet18.layer1(x)
        c3 = self.resnet18.layer2(c2)
        c4 = self.resnet18.layer3(c3)
        c5 = self.resnet18.layer4(c4)
        output = F.avg_pool2d(c5, c5.size()[2:])
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        return output