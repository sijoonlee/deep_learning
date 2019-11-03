## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        # input data 
        # image 1*224*224
        # label 68 * 2
        
        # Conv2d(in-channel, out-channel, kernel_size, stride=1, padding=0, dilation=1)
        # MaxPool2d(kernel_size, stride=None, padding=0, dilation=1)
        # Hout = ( Hin + 2 X padding - dilation X (kernel_size -1 ) - 1 ) / stride + 1
        self.conv1 = nn.Conv2d(1, 32, 5) 
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 64, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*50*50,136)
        #self.fc2 = nn.Linear(3200,136)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # input = batch * 1 * 224 * 224
        batch_size = x.shape[0]
        x = F.leaky_relu(self.conv1(x)) # Hout = 224 - 4 - 1 + 1 = 220
        x = F.leaky_relu(self.conv2(x)) # Hout = 220 - 4 = 216
        x = self.dropout(x)
        x = self.maxpool(x) # Hout = (216 - 1 - 1) / 2 + 1 = 108
        x = F.leaky_relu(self.conv3(x)) # Hout = 108 - 4 = 104
        x = F.leaky_relu(self.conv4(x)) # Hout = 104 - 4 = 100
        x = self.maxpool(x) # Hout = (200 - 1 - 1) / 2 + 1 = 50
        x = self.dropout(x)
        # size : batch * 64 * 50 * 50
        
        x = x.view(-1, 64*50*50)
        #x = F.leaky_relu(self.fc1(x))
        x = self.fc1(x) # ensure x has range 0 to 1
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
