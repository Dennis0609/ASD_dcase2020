import torch.nn as nn
import torch


def firstconv(in_channel, out_channel):
    return nn.Conv1d(in_channel, out_channel, 256, 128, 64)


def conv3x3(in_channel, out_channel, stride):
    net = nn.Sequential(
        nn.Conv1d(in_channel, out_channel, 3, stride, 1),
        nn.BatchNorm1d(out_channel),
        nn.LeakyReLU()
    )
    return net


def conv1x1(in_channel, out_channel):
    net = nn.Sequential(nn.Conv1d(in_channel, out_channel, 1, 1),
                        nn.BatchNorm1d(out_channel),
                        nn.LeakyReLU())
    return net


class block(nn.Module):
    def __init__(self, in_channel, out_channel, stride, factor):
        super(block, self).__init__()
        mid_channel = in_channel*factor
        self.stride = stride
        self.layer = nn.Sequential(
            conv1x1(in_channel, mid_channel),
            conv3x3(mid_channel, mid_channel, stride),
            conv1x1(mid_channel, out_channel) 
                                   )
        self.shortcut = nn.Sequential(
            conv1x1(in_channel, out_channel)
        )

    def forward(self, x):
        out = self.layer(x)
        out = (out+self.shortcut(x)) if self.stride == 1 else out
        return out


class Timenet(nn.Module):
    def __init__(self):
        super(Timenet, self).__init__()
        self.layer1 = firstconv(1, 16)
        self.layer2 = self.make_layer(16, 32, 1, 3)
        self.layer3 = self.make_layer(32, 32, 2, 3)
        self.layer4 = self.make_layer(32, 32, 2, 3)
        self.layer5 = self.make_layer(32, 64, 3, 2)
        self.layer6 = self.make_layer(64, 128, 2, 2)
        self.layer7 = self.make_layer(128, 256, 1, 2)
        self.layer8 = self.make_layer(256, 256, 2, 2)
        self.layer9 = self.make_layer(256, 256, 2, 2)
        self.layer10 = self.make_layer(256, 256, 2, 2)
        self.layer11 = self.make_layer(256, 512, 1, 2)
        self.avgpool = nn.AvgPool1d(kernel_size=7, stride=1)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, in_channel, out_channel, stride, in_factor):
        layers = []
        layers.append(block(in_channel, out_channel, stride, in_factor))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def get_number_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

