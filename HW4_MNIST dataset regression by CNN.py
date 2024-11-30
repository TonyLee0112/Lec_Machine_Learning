import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Set up Hyper Parameter
batch_size = 128
epochs = 30
learning_rate = 0.001


# Fix Random Seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

train_dataset = torchvision.datasets.MNIST(root="MNIST_data/", train=True, transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root="MNIST_data/",train=False,transform=transforms.ToTensor(),download=True)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.cnn = nn.Sequential(
      # nn. 인가 self. 인가
      nn.Conv2d(1,32,3,1,1),
      nn.ReLU(),
      nn.MaxPool2d(2,2),
      nn.Conv2d(32,64,3,1,1),
      nn.ReLU(),
      nn.MaxPool2d(2,2),
      nn.Flatten(),
      nn.Linear(3136,128),
      nn.Linear(128,10)
    )

  def forward(self,input):
    return self.cnn(input)

# Use GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Model Training
CNNmodel = CNN().to(device)
optimizer = optim.Adam(CNNmodel.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
  CNNmodel.train()
  avg_cost = 0
  total_batch_num = len(train_dataloader)
  for inputImage, label in train_dataloader:
    inputImage = inputImage.to(device)
    logits = CNNmodel(inputImage) # Logistic
    loss = criterion(logits, label.to(device)) # Get Loss

    avg_cost += loss / total_batch_num
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  print('Epoch : {} / {}, cost : {}'.format(epoch+1, epochs, avg_cost))

# Model Testing
correct = 0
total = 0

CNNmodel.eval()
for inputImage, label in test_dataloader:
  inputImage = inputImage.to(device)
  with torch.no_grad():
    logits = CNNmodel(inputImage)

  probs = nn.Softmax(dim = 1)(logits)
  predicts = torch.argmax(logits, dim = 1)

  total += len(label)
  correct += (predicts == label.to(device)).sum().item()

print(f'Accuracy of the network on test images : {100 * correct // total}%')

