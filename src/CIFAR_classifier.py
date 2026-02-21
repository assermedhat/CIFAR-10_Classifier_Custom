import torch
import torchvision
import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,32,3,padding="same")
        self.BN1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,32,3,padding="same")
        self.BN2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,64,3,padding="same")
        self.BN3=nn.BatchNorm2d(64)
        # self.conv4=nn.Conv2d(64,64,3)
        # self.BN4=nn.BatchNorm1d(64)
        self.fc1=nn.Linear(4*4*64,512)
        self.fc2=nn.Linear(512,128)
        self.fc3=nn.Linear(128,10)
        self.pool=nn.MaxPool2d(2)
        self.dropout=nn.Dropout(p=0.5)
    
    def forward(self,inputs):
        #layer 1 (Conv -> BN -> Activation -> pool)
        x=self.conv1(inputs)
        x=F.relu(self.BN1(x))
        x=self.pool(x)
        #layer 2
        x=self.conv2(x)
        x=F.relu(self.BN2(x))
        x=self.pool(x)
        #layer 3
        x=self.conv3(x)
        x=F.relu(self.BN3(x))
        x=self.pool(x)
        #flattened to (4x4x64)
        x=torch.flatten(x,1)
        #layer 4
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        #layer 5
        x=F.relu(self.fc2(x))
        x=self.dropout(x)
        #layer 6
        x=self.fc3(x)
        return x

class Initialize:
    def __init__(self,batch_size=64,):
        self.writer=SummaryWriter()
        self.batch_size=batch_size
        self.train_transforms=transforms.Compose([
            transforms.RandomCrop(32,padding=2),
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
        
        self.train_data=datasets.CIFAR10(
            root="data",
            train=True,
            transform = self.train_transforms,
            download=True
        )
        
        self.test_data=datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=self.test_transforms
        )

        self.train_dataloader=DataLoader(self.train_data,
                                         batch_size=batch_size,
                                         shuffle=True)

        self.test_dataloader=DataLoader(self.test_data,
                                        batch_size=batch_size,
                                        shuffle=True)
        
    def start(self):
        print(f"Training set size = {len(self.train_data)}\nTest set size= {len(self.test_data)}")
        print(f"total no of batches = {len(self.train_dataloader)}")
        return self.train_dataloader,self.test_dataloader,self.writer

class Orchestrator:
    def __init__(self,writer, train_dataloader,test_dataloader,model,lr=0.001,l2_reg=0,retrain=False,checkpoint_path="CIFAR.pth",batch_size=64):
        self.train_dataloader=train_dataloader
        self.test_dataloader=test_dataloader
        self.model=model
        self.retrain=retrain
        self.loss_fn=nn.CrossEntropyLoss()
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=lr,weight_decay=l2_reg)
        self.chkpt=checkpoint_path
        self.metric=torchmetrics.Accuracy(task="multiclass",num_classes=10).to(device)
        self.writer=writer
    
    def train_one_epoch(self):
        self.model.train()
        self.metric.reset()
        running_loss=0.0
        for train_input,train_labels in self.train_dataloader:
            train_input,train_labels=train_input.to(device),train_labels.to(device)
            self.opt.zero_grad()
            logits=self.model(train_input)
            loss=self.loss_fn(logits,train_labels)
            loss.backward()
            self.opt.step()
            self.metric.update(logits,train_labels)
            running_loss+=loss.item()
        avg_epoch_loss=running_loss/len(self.train_dataloader)
        return avg_epoch_loss,self.metric.compute().item()


    def evaluate(self):
        self.model.eval()
        self.metric.reset()
        running_loss=0.0
        with torch.no_grad():
            for test_input,test_labels in self.test_dataloader:
                test_input,test_labels=test_input.to(device),test_labels.to(device)
                logits=self.model(test_input)
                loss=self.loss_fn(logits,test_labels)
                running_loss+=loss.item()
                self.metric.update(logits,test_labels)
        avg_test_loss=running_loss/len(self.test_dataloader)
        return avg_test_loss,self.metric.compute().item()
    
    
    def fit(self,epochs=15):
        if not self.retrain:
            self.model.load_state_dict(torch.load(self.chkpt,map_location=device))
            print("Model already trained and ready for evaluation")
            return
        for epoch in range(epochs):
            train_loss,train_acc=self.train_one_epoch()
            test_loss,test_acc=self.evaluate()

            self.writer.add_scalars("Loss",
                                   {
                                       "train":train_loss,
                                       "test":test_loss
                                   },
                                   global_step=epoch)
            self.writer.add_scalars("Accuracy",
                                   {
                                       "train":train_acc,
                                       "test":test_acc
                                   },
                                   global_step=epoch)

            print(f"Epoch {epoch+1}/{epochs} : train_loss : {train_loss:.4f} | test_loss : {test_loss:.4f}")

        torch.save(self.model.state_dict(),self.chkpt)


    def log_test_preds(self, num_imgs=10, step=0):
        self.model.eval()

        images, labels = next(iter(self.test_dataloader))
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            logits = self.model(images)
            preds = logits.argmax(1)

        images = images.cpu()
        preds = preds.cpu()
        labels = labels.cpu()

        # CIFAR-10 normalization values
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

        fig = plt.figure(figsize=(num_imgs * 2, 3))

        for i in range(num_imgs):
            ax = fig.add_subplot(1, num_imgs, i + 1)

            img = images[i] * std + mean          # unnormalize
            img = img.permute(1, 2, 0).numpy()    # CHW → HWC

            ax.imshow(img)
            ax.set_title(f"P:{preds[i].item()} A:{labels[i].item()}")
            ax.axis("off")

        self.writer.add_figure("Test Predictions", fig, global_step=step)
    




if __name__ == "__main__":
    model=CNN().to(device)
    data=Initialize()
    train_dataloader,test_dataloader,writer=data.start()
    orch=Orchestrator(writer,train_dataloader,test_dataloader,model,lr=0.001,retrain=True,l2_reg=0.0009)
    orch.fit(epochs=100)
    test_loss,test_acc=orch.evaluate()
    print(f"Final test loss= {test_loss:.4f}\nFinal test acc = {test_acc*100:0.2f}")
    orch.log_test_preds()
    # print(f"Model Summary : {summary(model,(64,3,32,32))}")
    
