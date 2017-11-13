from __future__ import print_function
import argparse
import torch # pytorch package, allows using GPUs
from torch.autograd import Variable # differentiation of pytorch tensors
import torch.nn as nn # construct NN
import torch.nn.functional as F # implements forward and backward definitions of an autograd operation
import torch.optim as optim # different update rules such as SGD, Nesterov-SGD, Adam, RMSProp, etc
from torchvision import datasets, transforms # load MNIST data


#####################################################################################################################
###  https://github.com/pytorch/examples/tree/master/mnist                                                        ###
###  http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py  ###
#####################################################################################################################


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# set seed of random number generator
torch.manual_seed(args.seed)

kwargs = {} # CUDA arguments, if enabled
# load and noralise train and test data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        # inherit attributes and methods of nn.Module
        super(Net, self).__init__()

        # 1 input image channel, 10 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) 
        # 10 input image channels, 20 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # apply Dropout, default is p=0.5
        self.conv2_drop = nn.Dropout2d()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20*4*4, 50) # see forward function for dimensions
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        '''Defines the feed-forward function for the NN.

        A backward function is automatically using `torch.autograd`

        Parameters
        ----------
        x : autograd.Variable
            input data

        Returns
        -------
        autograd.Variable
            Output layer of NN
        
        '''

        # initial shape: x.size() = (*,1,28,28)
        # apply convolutional layer 1
        x = self.conv1(x) # x.size() = (*,10,24,24)
        # max pooling over a (2, 2) window
        x = F.max_pool2d(x, (2, 2)) # x.size() = (*,10,12,12)
        # apply relu
        x = F.relu(x) # x.size() = (*,1,12,12)
        # apply convolutional layer 2
        x = self.conv2(x) # x.size() = (*,20,8,8)
        # apply 2d dropout
        x = self.conv2_drop(x) # x.size() = (*,20,8,8)
        # if the size is a square you can only specify a single integer for size of squere
        x = F.max_pool2d(x, 2) # x.size() = (*,20,4,4)
        # apply relu
        x = F.relu(x) # x.size() = (*,20,4,4)
        
        # reshape data 
        x = x.view(-1, 20*4*4) # x.reshape(-1,20*4*4) 
        # apply rectified linear unit
        x = F.relu(self.fc1(x))
        # apply Dropout
        x = F.dropout(x, training=self.training)
        # apply affine operation fc2
        x = self.fc2(x)

        # soft-max layer
        x = F.log_softmax(x)

        return x

# create model
model = Net()
# negative log-likelihood (nll) loss for training
criterion = F.nll_loss
# define SGD optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    '''Trains a NN using minibatches.

    Parameters
    ----------
    epoch : int
        Training epoch number.

    '''

    # set model to training mode (affects Dropout and BatchNorm)
    model.train()
    # loop over training data
    for batch_idx, (data, label) in enumerate(train_loader):
        # wrap minibatch data in Variable
        data, label = Variable(data), Variable(label)
        # zero gradient buffers
        optimizer.zero_grad()
        # compute output of final layer: forward step
        output = model(data)
        # compute loss
        loss = criterion(output, label)
        # run backprop: backward step
        loss.backward()
        # update weigths of NN
        optimizer.step()
        # print loss at current epoch
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    '''Tests NN performance.

    '''

    # evaluate model
    model.eval()

    test_loss = 0 # loss function on test data
    correct = 0 # number of correct predictions
    # loop over test data
    for data, label in test_loader:
        # wrap test data in Variable
        data, label = Variable(data, volatile=True), Variable(label)
        # compute model prediction softmax probability
        output = model(data)
        # compute test loss
        test_loss += criterion(output, label, size_average=False).data[0] # sum up batch loss
        # find most likely prediction
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # update number of correct predictions
        correct += pred.eq(label.data.view_as(pred)).cpu().sum()

    # print test loss
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()