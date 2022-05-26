""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from U_net.unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = nn.Sequential(DoubleConv(n_channels, 64), nn.BatchNorm2d(64))
        self.down1 = nn.Sequential(Down(64, 128), nn.BatchNorm2d(128))
        self.down2 = nn.Sequential(Down(128, 256), nn.BatchNorm2d(256))
        self.down3 = nn.Sequential(Down(256, 512), nn.BatchNorm2d(512))
        factor = 2 if bilinear else 1
        self.down4 = nn.Sequential(Down(512, 1024 // factor), nn.BatchNorm2d(1024 // factor))
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Sequential(OutConv(64, n_classes), nn.BatchNorm2d(n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    # A full forward pass
    im = torch.randn(1, 3, 512, 512)
    model = UNet(3, 2)
    x = model(im)
    print(x.shape)
