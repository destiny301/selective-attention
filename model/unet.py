from model.unet_blocks import *

class UNet(nn.Module):
    """
    Original U-Net
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        x = nn.Sigmoid()(logits)
        return x

class ULite(nn.Module):
    """"
    U-Net-Lite(simplified U-Net)
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ULite, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 32)
        self.down3 = Down(32, 32)
        factor = 2 if bilinear else 1
        self.down4 = Down(32, 64 // factor)
        self.up1 = SingleUp(64, 64 // factor, bilinear)
        self.up2 = SingleUp(64, 64 // factor, bilinear)
        self.up3 = SingleUp(64, 32 // factor, bilinear)
        self.up4 = SingleUp(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # x(1, 320, 180)

        # encoder
        x1 = self.inc(x) # x1(16, 320, 180)
        x2 = self.down1(x1) # x2(32, ...)
        x3 = self.down2(x2) # x3(32, ...)
        x4 = self.down3(x3) # x4(32, ...)
        x5 = self.down4(x4) # x5(32, ...)

        # decoder
        x = self.up1(x5, x4) # x(32,...)
        x = self.up2(x, x3) # x(32, ...)
        x = self.up3(x, x2) # x(16, ...)
        x = self.up4(x, x1)
        logits = self.outc(x)
        x = nn.Sigmoid()(logits)
        return x