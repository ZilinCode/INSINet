# Auxiliary code for INSINet
# Code Related to MobileNet_v3 in INSINet


import torch
import torch.nn as nn
import torch.nn.functional as F

def Hswish(x,inplace=True):
    return x * F.relu6(x + 3., inplace=inplace) / 6.

def Hsigmoid(x,inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.



class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y=self.avg_pool(x).view(b, c)
        y=self.se(y)
        y = Hsigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,exp_channels,stride,se='True',nl='HS'):
        super(Bottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride=stride
        if se:
            self.se=SEModule(exp_channels)
        else:
            self.se=None
        self.conv1=nn.Conv2d(in_channels,exp_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2=nn.Conv2d(exp_channels,exp_channels,kernel_size=kernel_size,stride=stride,
                             padding=padding,groups=exp_channels,bias=False)
        self.bn2=nn.BatchNorm2d(exp_channels)
        self.conv3=nn.Conv2d(exp_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):
        out=self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out=self.bn2(self.conv2(out))
            out=self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV3_large(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg=[
        (16,3,16,1,False,'RE'),
        (24,3,64,2,False,'RE'),
        (24,3,72,1,False,'RE'),
        (40,5,72,2,True,'RE'),
        (40,5,120,1,True,'RE'),
        (40,5,120,1,True,'RE'),
        (80,3,240,2,False,'HS'),
        (80,3,200,1,False,'HS'),
        (80,3,184,1,False,'HS'),
        (80,3,184,1,False,'HS'),
        (112,3,480,1,True,'HS'),
        (112,3,672,1,True,'HS'),
        (160,5,672,2,True,'HS'),
        (160,5,960,1,True,'HS'),
        (160,5,960,1,True,'HS')
    ]
    def __init__(self,num_classes=17):
        super(MobileNetV3_large,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_channels=16)
        self.conv2=nn.Conv2d(160,960,1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(960)
        self.conv3=nn.Conv2d(960,1280,1,1,padding=0,bias=True)
        self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)

    def _make_layers(self,in_channels):
        layers=[]
        for out_channels,kernel_size,exp_channels,stride,se,nl in self.cfg:
            layers.append(
                Bottleneck(in_channels,out_channels,kernel_size,exp_channels,stride,se,nl)
            )
            in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=Hswish(self.bn2(self.conv2(out)))
        out=F.avg_pool2d(out,7)
        out=Hswish(self.conv3(out))
        out=self.conv4(out)
        a,b=out.size(0),out.size(1)
        out=out.view(a,b)
        return out

class MobileNetV3_small(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg = [
        (16,3,16,2,True,'RE'),
        (24,3,72,2,False,'RE'),
        (24,3,88,1,False,'RE'),
        (40,5,96,2,True,'HS'),
        (40,5,240,1,True,'HS'),
        (40,5,240,1,True,'HS'),
        (48,5,120,1,True,'HS'),
        (48,5,144,1,True,'HS'),
        (96,5,288,2,True,'HS'),
        (96,5,576,1,True,'HS'),
        (96,5,576,1,True,'HS')
    ]
    def __init__(self,num_classes=17):
        super(MobileNetV3_small,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        self.layers = self._make_layers(in_channels=16)
        self.conv2=nn.Conv2d(96,576,1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(576)
        self.conv3=nn.Conv2d(576,1280,1,1,padding=0,bias=True)
        self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)

    def _make_layers(self,in_channels):
        layers=[]
        for out_channels,kernel_size,exp_channels,stride,se,nl in self.cfg:
            layers.append(
                Bottleneck(in_channels,out_channels,kernel_size,exp_channels,stride,se,nl)
            )
            in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=self.bn2(self.conv2(out))
        se=SEModule(out.size(1))
        out=Hswish(se(out))
        out = F.avg_pool2d(out, 7)
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        a, b = out.size(0), out.size(1)
        out = out.view(a, b)
        return out
