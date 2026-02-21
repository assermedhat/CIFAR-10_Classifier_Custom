import torch.nn as nn
import torchvision


class ResNet18(nn.Module):
    def __init__(self,pretrained:bool):
        super().__init__()
        self.pretrained=pretrained
        if self.pretrained:
            self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            self.model = torchvision.models.resnet18(weights=None)
            
        #freezing model weights
        for param in self.model.parameters():
            param.requires_grad = False
        
        for p in self.model.layer4.parameters():
            p.requires_grad=True
        in_ftrs=self.model.fc.in_features
        self.model.fc=nn.Sequential(
            nn.Linear(in_ftrs,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,10))
        self.model.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        self.model.maxpool=nn.Identity()
    def forward(self,input):
        logits=self.model(input)
        return logits