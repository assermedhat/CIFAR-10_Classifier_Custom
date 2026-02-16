import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision import transforms
from CIFAR_classifier import Orchestrator,device
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
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

class Initialize_ResNet:
    def __init__(self):
        self.writer=SummaryWriter()
        self.train_transforms=transforms.Compose([
            transforms.RandomCrop(32,padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124),
                                 std=(0.24703233,0.24348505, 0.26158768))
        ])
        self.test_transforms=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124),
                                 std=(0.24703233,0.24348505, 0.26158768))
        ])

        self.train_data=datasets.CIFAR10(root="data",train=True,transform=self.train_transforms)
        self.test_data=datasets.CIFAR10(root="data",train=False,transform=self.test_transforms)

        self.train_dataloader=DataLoader(self.train_data,
                                         batch_size=64,
                                         shuffle=True)
        
        self.test_dataloader=DataLoader(self.test_data,
                                        batch_size=64,
                                        shuffle=True)
        
    def start(self):
        return self.train_dataloader,self.test_dataloader,self.writer
        

class Main(Orchestrator):
    def __init__(self,writer,train_dataloader,test_dataloader,model,lr=0.001,l2_reg=0.0,retrain=False,checkpoint="resnet_cifar10.pth"):
        super().__init__(writer,train_dataloader,test_dataloader,model=model,lr=lr,l2_reg=l2_reg,retrain=retrain,checkpoint_path=checkpoint)
        
        self.opt=torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=l2_reg)
        
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=30)
    def train_one_epoch(self):
        loss,acc=super().train_one_epoch()
        self.scheduler.step()
        return loss,acc




if __name__ == "__main__":
    model = ResNet18().to(device)
    data=Initialize_ResNet()
    train_data,test_data,writer=data.start()
    main=Main(writer,train_data,test_data,model=model,lr=0.1,retrain=False,l2_reg=0.0005)
    main.fit(epochs=30)
    test_loss,test_acc=main.evaluate()
    print(f"Final test loss = {test_loss:.4f}\nFinal test accuracy = {test_acc*100:.2f}")
    # summary(model,input_size=(64,3,32,32))