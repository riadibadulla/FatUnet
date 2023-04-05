from torch.nn.modules.pooling import MaxPool2d
import torch.nn as nn
import torch.nn.functional as F
import torch
from FatConv2d import FatConv2d
from torchsummary import summary

def double_conv_block(in_channels, out_channels, pool=True):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool:
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

def conv_layer(in_channels, out_channels, kernel_size=3):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same"),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = double_conv_block(1,64, pool=False)
        self.conv2 = double_conv_block(64,128, pool=False)
        self.conv3 = double_conv_block(128,256, pool=False)
        self.conv4 = double_conv_block(256,512, pool=False)
        self.conv5 = double_conv_block(512,1024, pool=False)
        self.deconv1= nn.ConvTranspose2d(1024,512, kernel_size=2, stride=2)
        self.conv6= double_conv_block(1024,512,pool=False)
        self.deconv2= nn.ConvTranspose2d(512,256, kernel_size=2, stride=2)
        self.conv7= double_conv_block(512,256,pool=False)
        self.deconv3= nn.ConvTranspose2d(256,128, kernel_size=2, stride=2)
        self.conv8= double_conv_block(256,128,pool=False)
        self.deconv4= nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
        self.conv9= double_conv_block(128,64,pool=False)
        self.segmenter = nn.Conv2d(64,1,kernel_size=1)

    def forward(self, x):


        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x = self.conv5(self.maxpool(x4))
        x = self.deconv1(x)
        x = self.conv6(torch.cat((x4,x),dim=1))
        x = self.deconv2(x)
        x = self.conv7(torch.cat((x3,x),dim=1))
        x = self.deconv3(x)
        x = self.conv8(torch.cat((x2,x),dim=1))
        x = self.deconv4(x)
        x = self.conv9(torch.cat((x1,x),dim=1))
        x = self.segmenter(x)
        return x

def double_conv_block(in_channels, out_channels, pool=True):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool:
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

def conv_layer(in_channels, out_channels, kernel_size=3):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same"),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

class UNet_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = double_conv_block(1,64, pool=False)
        self.conv2 = double_conv_block(64,128, pool=False)
        self.conv3 = double_conv_block(128,256, pool=False)
        self.conv4 = double_conv_block(256,512, pool=False)
        self.dropout = nn.Dropout(0.5)
        self.deconv1= nn.ConvTranspose2d(512,256, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.conv7= double_conv_block(512,256,pool=False)
        self.deconv2= nn.ConvTranspose2d(256,128, kernel_size=2, stride=2)
        self.conv8= double_conv_block(256,128,pool=False)
        self.deconv3= nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
        self.conv9= double_conv_block(128,64,pool=False)
        self.segmenter = nn.Conv2d(64,1,kernel_size=1)

    def forward(self, x):


        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x = self.dropout(x3)
        x = self.conv4(self.maxpool(x))
        # x = self.relu(self.deconv1(x))
        x = self.deconv1(x)
        x = self.conv7(torch.cat((x3,x),dim=1))
        x = self.deconv2(x)
        x = self.conv8(torch.cat((x2,x),dim=1))
        x = self.deconv3(x)
        x = self.conv9(torch.cat((x1,x),dim=1))
        x = self.segmenter(x)
        return x


class UNet_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = double_conv_block(1,64, pool=False)
        self.conv2 = double_conv_block(64,128, pool=False)
        self.conv3 = double_conv_block(128,256, pool=False)
        self.conv4 = double_conv_block(256,512, pool=False)
        self.conv5 = double_conv_block(512,1024, pool=False)
        self.dropout = nn.Dropout(0.5)
        self.deconv1= nn.ConvTranspose2d(1024,512, kernel_size=2, stride=2)
        self.conv6= double_conv_block(1024,512,pool=False)
        self.deconv2= nn.ConvTranspose2d(512,256, kernel_size=2, stride=2)
        self.conv7= double_conv_block(512,256,pool=False)
        self.deconv3= nn.ConvTranspose2d(256,128, kernel_size=2, stride=2)
        self.conv8= double_conv_block(256,128,pool=False)
        self.deconv4= nn.ConvTranspose2d(128,64, kernel_size=2, stride=2)
        self.conv9= double_conv_block(128,64,pool=False)
        self.segmenter = nn.Conv2d(64,1,kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x = self.dropout(x4)
        x = self.conv5(self.maxpool(x))
        x = self.dropout(x)
        x = self.deconv1(x)
        x = self.conv6(torch.cat((x4,x),dim=1))
        x = self.deconv2(x)
        x = self.conv7(torch.cat((x3,x),dim=1))
        x = self.deconv3(x)
        x = self.conv8(torch.cat((x2,x),dim=1))
        x = self.deconv4(x)
        x = self.conv9(torch.cat((x1,x),dim=1))
        x = self.segmenter(x)
        return x

class adapted_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = double_conv_block(3,64, pool=False)
        self.conv2 = double_conv_block(64,128, pool=False)
        self.conv3 = double_conv_block(128,256, pool=False)
        self.conv4 = double_conv_block(256,512, pool=False)
        self.conv5 = double_conv_block(512,1024, pool=False)
        self.deconv1= nn.Sequential(nn.Conv2d(1024,512, kernel_size=2, stride=1, padding="same"),nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv6= double_conv_block(1024,512,pool=False)
        self.deconv2= nn.Sequential(nn.Conv2d(512,256, kernel_size=2, stride=1, padding="same"),nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv7= double_conv_block(512,256,pool=False)
        self.deconv3= nn.Sequential(nn.Conv2d(256,128, kernel_size=2, stride=1, padding="same"),nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv8= double_conv_block(256,128,pool=False)
        self.deconv4= nn.Sequential(nn.Conv2d(128,64, kernel_size=2, stride=1, padding="same"),nn.Upsample(scale_factor=2, mode='nearest'))
        self.conv9= double_conv_block(128,64,pool=False)
        self.segmenter = nn.Conv2d(64,1,kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x = self.conv5(self.maxpool(x4))
        x = self.deconv1(x)
        x = self.conv6(torch.cat((x4,x),dim=1))
        x = self.deconv2(x)
        x = self.conv7(torch.cat((x3,x),dim=1))
        x = self.deconv3(x)
        x = self.conv8(torch.cat((x2,x),dim=1))
        x = self.deconv4(x)
        x = self.conv9(torch.cat((x1,x),dim=1))
        x = self.segmenter(x)
        return x

class contracting_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = double_conv_block(3,64, pool=False)
        self.conv2 = double_conv_block(64,128, pool=False)
        self.conv3 = double_conv_block(128,256, pool=False)
        self.conv4 = double_conv_block(256,512, pool=False)
        self.conv5 = double_conv_block(512,1024, pool=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x = self.conv5(self.maxpool(x4))
        # x = self.deconv1(x)
        # x = self.conv6(torch.cat((x4,x),dim=1))
        # x = self.deconv2(x)
        # x = self.conv7(torch.cat((x3,x),dim=1))
        # x = self.deconv3(x)
        # x = self.conv8(torch.cat((x2,x),dim=1))
        # x = self.deconv4(x)
        # x = self.conv9(torch.cat((x1,x),dim=1))
        # x = self.segmenter(x)
        return x

class fat_UNet_non_refined(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(conv_layer(1,32,kernel_size=5),conv_layer(32,32,kernel_size=6))
        self.conv2 = nn.Sequential(conv_layer(32,16,kernel_size=12),conv_layer(16,16,kernel_size=24))
        self.conv3 = nn.Sequential(conv_layer(16,8,kernel_size=48),conv_layer(8,8,kernel_size=96))
        self.conv4 = nn.Sequential(conv_layer(8,10,kernel_size=122),conv_layer(10,10,kernel_size=160))
        self.conv5 = nn.Sequential(conv_layer(10,20,kernel_size=154),conv_layer(20,20,kernel_size=160))
        self.dropout = nn.Dropout(0.5)
        self.deconv1= nn.Sequential(nn.Conv2d(20,10, kernel_size=3, stride=1, padding="same"))
        self.conv6= nn.Sequential(conv_layer(20,10,kernel_size=113),conv_layer(10,10,kernel_size=160))#kernels recalculate
        self.deconv2= nn.Sequential(nn.Conv2d(10,8, kernel_size=3, stride=1, padding="same"))
        self.conv7= nn.Sequential(conv_layer(16,8,kernel_size=48),conv_layer(8,8,kernel_size=96))
        self.deconv3= nn.Sequential(nn.Conv2d(8,16, kernel_size=3, stride=1, padding="same"))
        self.conv8= nn.Sequential(conv_layer(32,16,kernel_size=12),conv_layer(16,16,kernel_size=24))
        self.deconv4= nn.Sequential(nn.Conv2d(16,32, kernel_size=3, stride=1, padding="same"))
        self.conv9= nn.Sequential(conv_layer(64,32,kernel_size=3),conv_layer(32,3,kernel_size=5))
        self.segmenter = nn.Conv2d(3,1,kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.dropout(x4)
        x = self.conv5(x)
        x = self.dropout(x)
        x = self.deconv1(x)
        x = self.conv6(torch.cat((x4,x),dim=1))
        x = self.deconv2(x)
        x = self.conv7(torch.cat((x3,x),dim=1))
        x = self.deconv3(x)
        x = self.conv8(torch.cat((x2,x),dim=1))
        x = self.deconv4(x)
        x = self.conv9(torch.cat((x1,x),dim=1))
        x = self.segmenter(x)
        return x


#
# class FatU_Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv4_skip = conv_layer(512,64)
#         self.conv3_skip = nn.Sequential(conv_layer(256,64),conv_layer(64,16))
#         self.conv2_skip = nn.Sequential(conv_layer(128,32),conv_layer(32,8))
#         self.conv1_skip = nn.Sequential(conv_layer(64,16),conv_layer(16,4))
#
#         self.maxpool = nn.MaxPool2d(2)
#         self.conv1 = double_conv_block(3,64, pool=False)
#         self.conv2 = double_conv_block(64,128, pool=False)
#         self.conv3 = double_conv_block(128,256, pool=False)
#         self.conv4 = double_conv_block(256,512, pool=False)
#         self.conv5 = double_conv_block(512,1024, pool=False)
#         self.deconv1= FatConv2d(1024,6, kernel_size=20)
#
#         self.conv6= double_conv_block(70,64,pool=False)
#         self.deconv2= FatConv2d(64,6, kernel_size=40)
#         self.conv7= double_conv_block(22,16,pool=False)
#         self.deconv3= FatConv2d(16,2, kernel_size=80)
#         self.conv8= double_conv_block(10,8,pool=False)
#         self.deconv4= FatConv2d(8,2, kernel_size=160)
#         self.conv9= double_conv_block(6,4,pool=False)
#         self.segmenter = nn.Conv2d(4,1,kernel_size=1)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(self.maxpool(x1))
#         x3 = self.conv3(self.maxpool(x2))
#         x4 = self.conv4(self.maxpool(x3))
#         x = self.conv5(self.maxpool(x4))
#         x = self.deconv1(x)
#         x4 = self.conv4_skip(x4)
#         x3 = self.conv3_skip(x3)
#         x2 = self.conv2_skip(x2)
#         x1 = self.conv1_skip(x1)
#         x = self.conv6(torch.cat((x4,x),dim=1))
#         x = self.deconv2(x)
#         x = self.conv7(torch.cat((x3,x),dim=1))
#         x = self.deconv3(x)
#         x = self.conv8(torch.cat((x2,x),dim=1))
#         x = self.deconv4(x)
#         x = self.conv9(torch.cat((x1,x),dim=1))
#         x = self.segmenter(x)
#         return x
#
# class FatU_Net02beta(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv4_skip = conv_layer(512,64)
#         self.conv3_skip = nn.Sequential(conv_layer(256,64),conv_layer(64,16))
#         self.conv2_skip = nn.Sequential(conv_layer(128,32),conv_layer(32,4))
#         self.conv1_skip = nn.Sequential(conv_layer(64,16),conv_layer(16,4))
#
#         self.maxpool = nn.MaxPool2d(2)
#         self.conv1 = double_conv_block(3,64, pool=False)
#         self.conv2 = double_conv_block(64,128, pool=False)
#         self.conv3 = double_conv_block(128,256, pool=False)
#         self.conv4 = double_conv_block(256,512, pool=False)
#         self.conv5 = double_conv_block(512,1024, pool=False)
#         self.deconv1= FatConv2d(1024,64, kernel_size=20)
#
#         self.conv6= double_conv_block(128,64,pool=False)
#         self.deconv2= FatConv2d(64,16, kernel_size=40)
#         self.conv7= double_conv_block(32,16,pool=False)
#         self.deconv3= FatConv2d(16,4, kernel_size=80)
#         self.conv8= double_conv_block(8,4,pool=False)
#         self.deconv4= FatConv2d(4,1, kernel_size=160)
#         self.conv9= double_conv_block(5,4,pool=False)
#         self.segmenter = nn.Conv2d(4,1,kernel_size=1)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(self.maxpool(x1))
#         x3 = self.conv3(self.maxpool(x2))
#         x4 = self.conv4(self.maxpool(x3))
#         x = self.conv5(self.maxpool(x4))
#         x = self.deconv1(x)
#         x4 = self.conv4_skip(x4)
#         x3 = self.conv3_skip(x3)
#         x2 = self.conv2_skip(x2)
#         x1 = self.conv1_skip(x1)
#         x = self.conv6(torch.cat((x4,x),dim=1))
#         x = self.deconv2(x)
#         x = self.conv7(torch.cat((x3,x),dim=1))
#         x = self.deconv3(x)
#         x = self.conv8(torch.cat((x2,x),dim=1))
#
#         x = self.deconv4(x)
#         x = self.conv9(torch.cat((x1,x),dim=1))
#         x = self.segmenter(x)
#         return x

def test():
    # x = torch.randn((3, 3, 160, 160), requires_grad=False)
    # model = adapted_UNet()
    model = fat_UNet_non_refined()
    # print(model)
    # from FatSpitter import FatSpitter
    # model = FatSpitter.fat_spitter(model,optical=False)
    # print("all done")
    # print(model)
    summary(model.to(torch.device("cuda")),(3,160,160),device="cuda")


if __name__ == "__main__":
    test()