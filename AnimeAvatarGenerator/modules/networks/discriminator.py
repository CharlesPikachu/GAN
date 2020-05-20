'''
Function:
    define the discriminator
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch
import torch.nn as nn


'''discriminator'''
class Discriminator(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64*2),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64*2, out_channels=64*4, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64*4),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=64*4, out_channels=64*8, kernel_size=4, stride=2, padding=1, bias=False),
                                   nn.BatchNorm2d(64*8),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=64*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
                                   nn.Sigmoid())
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x.view(batch_size, -1)