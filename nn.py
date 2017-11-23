
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import pre_process as pp

inputs, outputs = pp.get_data()


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.layer1 = nn.Linear(6, 6)
        self.layer2 = nn.Linear(6, 12)
        self.layer3 = nn.Linear(12, 12)
        self.layer4 = nn.Linear(12, 12)
        self.layer5 = nn.Linear(12, 6)
        self.layer6 = nn.Linear(6, 18)

    def forward(self, inputs):
        hidden_neurons = F.relu(self.layer1(inputs))
        hidden_neurons = F.relu(self.layer2(hidden_neurons))
        hidden_neurons = F.relu(self.layer3(hidden_neurons))
        hidden_neurons = F.relu(self.layer4(hidden_neurons))
        hidden_neurons = F.relu(self.layer5(hidden_neurons))
        output_neurons = F.softmax(self.layer6(hidden_neurons))
        return output_neurons

model1 = Model()
model2 = Model()

model1.cuda()
model2.cuda()

try:
    model1.load_state_dict(torch.load('model1'))
    model2.load_state_dict(torch.load('model2'))
except Exception as e:
    print(e)

tensor = autograd.Variable(torch.cuda.FloatTensor(inputs))

target = autograd.Variable(torch.cuda.FloatTensor(outputs))

criterion = nn.MSELoss()

optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

for i in range(100000):
    optimizer1.zero_grad()
    output1 = model1(tensor)
    print(output1[1])
    loss1 = criterion(output1, target)
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    output2 = model2(tensor)
    print(output2[1])
    loss2 = criterion(output2, target)
    loss2.backward()
    optimizer2.step()

torch.save(model1.state_dict(), 'model1')
torch.save(model2.state_dict(), 'model2')
