import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class encoder(nn.Module):
    def __init__(self,latent_dim):
        super(encoder,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.lin1 = nn.Linear(64*7*7, 120)
        self.lin2 = nn.Linear(120,latent_dim)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        return x

class decoder(nn.Module):
    def __init__(self,latent_dim):
        super(decoder,self).__init__()
        self.lin1 = nn.Linear(latent_dim,120)
        self.lin2 = nn.Linear(120,64*7*7)
        self.conv1 = nn.ConvTranspose2d(64,32,3,2,1,output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 1, 3, 2, 1,output_padding=1)

    def forward(self,x):
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = x.view(x.size(0),64,7,7)
        x = torch.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

class VAE(nn.Module):
    def __init__(self,latent_dim):
        super(VAE,self).__init__()
        self.encode = encoder(latent_dim)
        self.decode = decoder(latent_dim)

    def forward(self,x):
        x = self.encode(x)
        recons = self.decode(x)
        return recons

def plot_result(orig, recons,epoch):
    orig = orig.cpu().data.numpy()
    recons = recons.cpu().data.numpy()

    orig = orig[:8].reshape(-1,28,28)
    recons = recons[:8].reshape(-1, 28, 28)
    fig, axis = plt.subplots(8,2)
    for i in range(8):
        axis[i,0].imshow(orig[i],cmap="gray")
        axis[i,0].axis("off")
        axis[i,1].imshow(recons[i], cmap="gray")
        axis[i,1].axis("off")

    plt.suptitle(f"Epoch: {epoch + 1}")
    plt.show()

def loss(orig,recons):
    calc_loss = nn.BCELoss(reduction="sum")(recons.view(-1,28*28), orig.view(-1,28*28))
    return calc_loss

def train_model(model, trainSet, lr = 0.01, epochs = 3, device = "cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_no, (data,_) in enumerate(trainSet):
            data = data.to(device)
            recons = model(data)
            l = loss(data,recons)
            total_loss += l.item()
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} Loss: {total_loss/len(trainSet.dataset)}")

        with torch.no_grad():
            model.eval()
            sample, _ = next(iter(trainSet))
            recons = model(sample)
            plot_result(sample,recons,epoch)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(20).to(device)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

train_model(model,train_loader,0.01,3)