
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, maxpool_size=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(maxpool_size),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=stride)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        #upconv x1 to match x2 size by padding then concatenate x1 and x2 for skip connection
        x1 = self.up(x1)
        # input is batch_size x n_channels x H x W
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, layer_dimensions=[64,128,256,512,1024], kernel_size=3, maxpool_size=2, upconv_stride=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, layer_dimensions[0], kernel_size=kernel_size))

        self.down = nn.ModuleList()
        for i in range(len(layer_dimensions)-1):
            self.down.append(Down(layer_dimensions[i], layer_dimensions[i+1], kernel_size=kernel_size, maxpool_size=maxpool_size))

        self.up = nn.ModuleList()
        for i in range(len(layer_dimensions)-1, 0, -1):
            self.up.append(Up(layer_dimensions[i], layer_dimensions[i-1], kernel_size=kernel_size-1, stride=upconv_stride))

        self.outc = (OutConv(layer_dimensions[0], n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        layer_output_list = [x1]
        for i, down in enumerate(self.down):
            layer_output_list.append(down(layer_output_list[-1]))
        x = layer_output_list[-1]
        for i, up in enumerate(self.up):
            x = up(x, layer_output_list[-2-i])
        output = self.outc(x)
        return output