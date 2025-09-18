import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms 

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])

])


train_dir = '~/datasets/fruits/fruits-360_100x100/fruits-360/Training'
test_dir  = '~/datasets/fruits/fruits-360_100x100/fruits-360/Test'


train_dataset = datasets.ImageFolder(train_dir, transform = transform)
test_dataset = datasets.ImageFolder(test_dir, transform = transform)

train_loader = DataLoader(train_dataset , batch_size = 64 ,shuffle=True) 
test_loader = DataLoader(test_dataset , batch_size = 64 ,shuffle=True)

num_classes = len(train_dataset.classes)

print(f"Number of classes :{num_classes}")

import torch.nn as nn
class SimpleCNN (nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN,self).__init__()
        self.conv1 = nn.Conv2d(3,16,kernel_size = 3, padding =1)
        self.conv2= nn.Conv2d(16,32,kernel_size = 3, padding =1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*16*16,128)
        self.fc2 = nn.Linear(128,num_classes)

    def forward(self,X):
        X = self.pool(F.relu(self.conv1(X)))
        X = self.pool(F.relu(self.conv2(X)))
        X = X.view(X.size(0),-1)
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        return X 
 
model = SimpleCNN(num_classes).to(device)       

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

num_epochs = 5 

for epoch in range(num_eporchs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
        images,labels = images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')


