from __future__ import print_function
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import numpy as np



use_cuda = False
momentum_par = 0.5
lr = 0.01
log_interval = 5400
epochs = 1000
batch_size = 64


loader_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

torch.manual_seed(42)

print('Loading Training Data')
train_data = pickle.load(open('data/generated_train_data_norm.p', 'rb'))
train_data = torch.from_numpy(train_data).float()#.resize_(27000,1,28,28)
train_label = pickle.load(open('data/generated_train_labels.p', 'rb'))
train_label = torch.from_numpy(train_label).long()

print('Loading Validation Data')
valid_data = pickle.load(open('data/generated_valid_data_norm.p', 'rb'))
valid_data = torch.from_numpy(valid_data).float().resize_(len(valid_data),1,28,28)

valid_label = pickle.load(open('data/generated_valid_labels.p', 'rb'))
valid_label = torch.from_numpy(valid_label).long()

print('Creating Data Loaders')
train_loader = DataLoader(TensorDataset(train_data, train_label),
                            batch_size = batch_size,
                            shuffle=True)
valid_loader = DataLoader(TensorDataset(valid_data, valid_label),
                            batch_size = 1,
                            shuffle=True)

# network given in assignment
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        return F.log_softmax(x)


model = Net()

if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum_par)
# optimizer = optim.Adam(model.parameters())
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)
# optimizer = optim.Adagrad(model.parameters())
# optimizer = optim.Adadelta(model.parameters())



def predict(model, x_val):
    x = Variable(x_val, requires_grad=False)
    output = model.forward(x)
    return output.data.numpy().argmax(axis=1)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target[:,0])
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.data[0]))



def test(epoch, valid_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in valid_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target[:,0])
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(valid_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(valid_loader.dataset),
        100. * correct / len(valid_loader.dataset)))

for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch, valid_loader)

# def test(epoch, valid_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in valid_loader:
#         if use_cuda:
#             data, target = data.cuda(), target.cuda()
#         data, target = Variable(data, volatile=True), Variable(target[:,0])
#         output = model(data)
#         test_loss += F.nll_loss(output, target).data[0]
#         pred = output.data.max(1)[1] # get the index of the max log-probability
#         correct += pred.eq(target.data).cpu().sum()

#     test_loss /= len(valid_loader) # loss function already averages over batch size
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(valid_loader.dataset),
#         100. * correct / len(valid_loader.dataset)))

# for epoch in xrange(0, epochs):
#     train(epoch)
#     # test(epoch, valid_loader)
#     predY = predict(model, valid_data)
#     pred_train_y = predict(model, train_data)
#     print("Epoch %d, train_acc = %.2f%% val_acc = %.2f%%"
#           % (epoch + 1, 
#             # cost / (n_examples/batch_size), 
#             100. * np.mean(pred_train_y == train_label.numpy()),
#             100. * np.mean(predY == valid_label.numpy())))
