import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from hyperparams import Hyperparams as hp
from data_load import load_data
from tqdm import tqdm
import math

print("loading data..")
X, Y = load_data()

def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-3):
    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)
    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)
    return lr_lambda

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer1 = nn.Conv2d(1, 512, 3, 1, 1)
        self.layer2 = nn.LeakyReLU() #F.relu(inplace=True)
        self.layer3 = nn.BatchNorm2d(512)
        new = []
        for i in range(20):
            new.append(nn.Conv2d(512, 512, 3, 1, 1))
            new.append(nn.BatchNorm2d(512))
            new.append(nn.LeakyReLU()) #F.relu(inplace=True))

        self.layer4 = nn.Conv2d(512, 10, 3, 1, 1)
        self.layer6 = nn.BatchNorm2d(10)
        self.layer5 = nn.LeakyReLU()#F.relu(inplace=True) #LeakyReLU()
#        self.layer6 = nn.BatchNorm2d(1)
        self.softmax = nn.Softmax()
        layers = [self.layer1, self.layer3,  self.layer2]+ new+ [ self.layer4, self.layer6, self.layer5, self.softmax]
#        softmax = nn.Softmax()
        self.net = nn.Sequential(*layers)

    def forward(self, x):

        out = self.net(x)
        return out     


cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm2d") != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

generator = Generator()

if cuda:
    generator.cuda()
#generator.apply(weights_init_normal)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

factor = 100
end_lr = lr_max = 3*10e-3
step_size = X.shape[0]
learning_rate = 1e-2
optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(generator.parameters(), lr=1.)
#clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer) #, [clr])
#scheduler.step()
#print(X, Y, X.shape, Y.shape)
#print(X.shape[0])

criterion = nn.CrossEntropyLoss().cuda()
batch = 64
iters = int(X.shape[0]/batch)
epochs = 100
for e in range(epochs):
  for i in range(iters):
    x_arr = []
    y_arr = []
    for j in range(batch):  
        x, y = X[i*batch+j], Y[i*batch+j]
        x_arr.append(torch.tensor(x).unsqueeze(0))
        y_arr.append(torch.tensor(y))
    y = torch.stack(y_arr)
#    y = y.unsqueeze(0)
    y = Variable(y.type(torch.cuda.LongTensor))
    x = torch.stack(x_arr)
    x = Variable(x.type(Tensor) ) #, requires_grad=True)
    select = torch.eq(x, torch.zeros_like(x)).type(Tensor).squeeze(1)
    y_hat = generator(x) #, requires_grad=True)

    preds, indices = torch.max(y_hat, 1)
    optimizer.zero_grad()
#    print(y_hat, y) 
#    loss = nn.MSELoss()
#    print(y.shape, y_hat.shape)
    loss = criterion(y_hat, y)
    
    loss_d = torch.sum(loss*select)/torch.sum(select)
    loss_d.backward()
    optimizer.step()

#    equal = torch.eq(indices.int(), y.int())
    equal = indices.int()==y.int()

    hits = select[equal] #.type(Tensor)
    print(hits.shape)
    accuracy = torch.sum(hits)/torch.sum(select)+1e-8
    print("loss:", loss_d.item(), "iteration:", i, "epoch:", e, "hits:", torch.sum(hits).item(),"/", torch.sum(select).item(),  "accuracy:", accuracy.item())

