import torch
import torch.nn as nn
import torch.nn.functional as F

###depth wise
def Conv3x3BNReLU(in_channels,out_channels,stride,groups):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
####expansion layer
def Conv1x1BNReLU(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

##projection layer
def Conv1x1BN(in_channels,out_channels):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels)
        )
'''
stride=1 和stride=2时结构不一样，步长为2没有shortcut
'''
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        mid_channels = (in_channels * expansion_factor)
        self.bottleneck = nn.Sequential(
            Conv1x1BNReLU(in_channels, mid_channels),
            Conv3x3BNReLU(mid_channels, mid_channels, stride, groups=mid_channels),
            Conv1x1BN(mid_channels, out_channels)
        )
        if self.stride == 1:
            self.shortcut = Conv1x1BN(in_channels, out_channels)
    def forward(self,x):
        out = self.bottleneck(x)
        out = (out+self.shortcut(x)) if self.stride==1 else out
        return out   

    
class Mobilenetv2(nn.Module):
    def __init__(self, t=6):
        super(Mobilenetv2,self).__init__()

        self.first_conv = Conv3x3BNReLU(2,32,2,groups=1)
        self.layer1 = self.make_layer(in_channels=32, out_channels=16, stride=1, factor=1, block_num=1)
        self.layer2 = self.make_layer(in_channels=16, out_channels=24, stride=2, factor=t, block_num=2)
        self.layer3 = self.make_layer(in_channels=24, out_channels=32, stride=2, factor=t, block_num=3)
        self.layer4 = self.make_layer(in_channels=32, out_channels=64, stride=2, factor=t, block_num=4)
        self.layer5 = self.make_layer(in_channels=64, out_channels=96, stride=1, factor=t, block_num=3)
        self.layer6 = self.make_layer(in_channels=96, out_channels=160, stride=2, factor=t, block_num=3)
        self.layer7 = self.make_layer(in_channels=160, out_channels=320, stride=1, factor=t, block_num=1)
        self.last_conv = Conv1x1BNReLU(320,1280)
        self.avgpool = nn.AvgPool2d(kernel_size=7,stride=1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

    def make_layer(self, in_channels, out_channels, stride, factor, block_num):
        layers = []
        layers.append(InvertedResidual(in_channels, out_channels, factor, stride))
        for i in range(1, block_num):
            layers.append(InvertedResidual(out_channels, out_channels, factor, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_conv(x)  
        x = self.layer1(x)      
        x = self.layer2(x)      
        x = self.layer3(x)      
        x = self.layer4(x)      
        x = self.layer5(x)      
        x = self.layer6(x)      
        x = self.layer7(x)      
        x = self.last_conv(x)  
        x = self.avgpool(x)     
        x = x.view(x.size(0),-1)    
        
        return x
    
    def get_number_parameters(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
