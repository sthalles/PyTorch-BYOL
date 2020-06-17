import torchvision.models as models
import torch

from models.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=False)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, **kwargs['projection_head'])

    def forward(self, x):
        h = self.encoder(x)
        return self.projetion(h.view(h.shape[0], h.shape[1]))