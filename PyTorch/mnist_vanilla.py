# -*- coding: utf-8 -*-
"""
Criado em Wed Mar 31 16:00:00 2021

@author: Jasmine Moreira

1) Preparar dados
2) Criar o modelo (input, output size, forward pass)
3) Criar a função de erro (loss) e o otimizador 
4) Criar o loop de treinamento
   - forward pass: calcular a predição e o erro
   - backward pass: calcular os gradientes
   - update weights: ajuste dos pesos do modelo
"""

import torch as t
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

use_cuda = t.cuda.is_available()
device = t.device("cuda:0" if use_cuda else "cpu") 

#Preparar dados

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = t.utils.data.DataLoader(mnist_trainset, batch_size=128, shuffle=False)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = t.utils.data.DataLoader(mnist_testset, batch_size=128, shuffle=False)

"""
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(28*28, 100) 
        self.linear2 = nn.Linear(100, 50) 
        self.final = nn.Linear(50, 10)

    def forward(self, img): #convert + flatten
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x
net = Net()
"""

# Criar Modelo
net = nn.Sequential(nn.Linear(28*28, 50),
                      nn.ReLU(),
                      nn.Linear(50, 10)
                      )
if use_cuda:
    net.cuda()

# Erro e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = t.optim.AdamW(net.parameters(), lr=0.01) #e-1


# loop de treinamento
epoch = 10

loss_values = []
for epoch in range(epoch):
    #net.train()
    epoch_loss = 0
    for data in train_loader:
        x, y = data
        optimizer.zero_grad()
        output = net(x.view(-1, 28*28).to(device)).to(device)
        loss = criterion(output, y.to(device))
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss+loss.item()       
    loss_values.append(epoch_loss/len(train_loader))
    print(epoch_loss/len(train_loader))
    
plt.plot(range(1,epoch+2), loss_values, 'bo', label='Training Loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

      
with t.no_grad():
    for data in test_loader:      
        x, y = data
        output = net(x.view(-1, 784).to(device))
        correct = 0
        total = 0
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct +=1
            total +=1
print(f'accuracy: {round(correct/total, 3)}')


"""
plt.imshow(x[3].view(28, 28))
plt.show()
print(t.argmax(net(x[3].view(-1, 784))[0]))
"""
