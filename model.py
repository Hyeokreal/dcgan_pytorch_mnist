import torch.nn as nn
import torch.nn.functional as F

# for test
import torch
from torch.autograd import Variable

z_dim = 100
x_dim = 64

batch = 100


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = list()
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    layers = list()
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)


def linear(l_in, l_out, bn=True):
    layers = list()
    layers.append(nn.Linear(l_in, l_out))
    if bn:
        layers.append(nn.BatchNorm1d(l_out))
    return nn.Sequential(*layers)

# gen relu
# gen tanh
class G(nn.Module):
    def __init__(self,d = 128):
        super(G, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    def forward(self, input):
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))
        return x


class D(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

'''
shape Test
'''

if __name__=="__main__":
    batch = 100
    z = torch.zeros([batch, z_dim])
    z = z.view([-1,z_dim, 1, 1])
    z = Variable(z)

    x = torch.zeros([batch, 1, 64, 64])
    x = Variable(x)

    g = G()
    out = g.forward(z)

    d = D()
    out2 = d.forward(x)

    r1 = out.data.numpy()
    r2 = out2.data.numpy()

    print(r1.shape)
    print(r2.shape)
