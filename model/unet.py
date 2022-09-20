from model.unet_blocks import *

class Encoder(nn.Module):
    """
    encoding features with conv
    """
    def __init__(self, n_channels, config, bilinear=True):
        super(Encoder, self).__init__()
        # cnn feature encoder
        self.inc = DoubleConv(n_channels, config[0])
        self.down1 = Down(config[0], config[1])
        self.down2 = Down(config[1], config[2])
        self.down3 = Down(config[2], config[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(config[3], config[4] // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x1, x2, x3, x4, x5

class Decoder(nn.Module):
    def __init__(self, n_classes, config, bilinear=True):
        super(Decoder, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(config[4], config[3] // factor, bilinear)
        self.up2 = Up(config[3], config[2] // factor, bilinear)
        self.up3 = Up(config[2], config[1] // factor, bilinear)
        self.up4 = Up(config[1], config[0], bilinear)
        self.outc = OutConv(config[0], n_classes)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet(nn.Module):
    """"
    1. encoding valid patches into features
    2. decoding the features
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        enc_config = [64, 128, 256, 512 ,1024]
        dec_config = [64, 128, 256, 512 ,1024]
        self.encoder = Encoder(n_channels, enc_config,  bilinear=bilinear)
        
        self.decoder = Decoder(n_classes, dec_config, bilinear=bilinear)
    


    def forward(self, x):
        # x(B, 3, 224, 224)
        # ecoder
        x1, x2, x3, x4, x5 = self.encoder(x)

        # decoder
        x = self.decoder(x1, x2, x3, x4, x5)

        return x

class ULite(nn.Module):
    """"
    U-Net-Lite(simplified U-Net)
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ULite, self).__init__()

        enc_config = [16, 32, 32, 32 ,64]
        dec_config = [16, 32, 64, 64 ,64]
        self.encoder = Encoder(n_channels, enc_config,  bilinear=bilinear)
        
        self.decoder = Decoder(n_classes, dec_config, bilinear=bilinear)
        

    def forward(self, x):
        # x(B, 3, 224, 224)
        # ecoder
        x1, x2, x3, x4, x5 = self.encoder(x)

        # decoder
        x = self.decoder(x1, x2, x3, x4, x5)

        return x