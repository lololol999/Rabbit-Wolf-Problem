import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# нейросеть и слои
class Net(nn.Module):
   def __init__(self):
       super(Net, self).__init__()
       self.fc1 = nn.Linear(5, 10)
       self.fc2 = nn.Linear(10, 10)
       self.fc3 = nn.Linear(10, 10)
       self.fc4 = nn.Linear(10, 6)
       self.fc5 = nn.Linear(6, 2)

   def forward(self, x):
       x = F.relu(self.fc1(x))
       x = F.relu(self.fc2(x))
       x = self.fc3(x)
       return F.log_softmax(x)

net = Net()
print(net)

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

criterion = nn.NLLLoss()

# тренировочный цикл: TODO