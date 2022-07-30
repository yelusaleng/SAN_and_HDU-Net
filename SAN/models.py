import math

from options_GAN import opt
from unet_parts import *


opt = opt()


class Predictor(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(Predictor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_channels)
        self.outc1 = OutConv(n_channels, n_classes)

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
        x = self.outc(x)
        logits = self.outc1(x)
        return x, F.sigmoid(logits)



def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) // 2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) // 2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)

    return spp


# class Discriminator(nn.Module):
#     def __init__(self, n_channels=3):
#         super(Discriminator, self).__init__()
#
#         self.output_num = [4, 2, 1]
#
#         self.model = nn.Sequential(
#             DoubleConv(n_channels, 64),
#             Down(64, 128),
#             Down(128, 256),
#             Down(256, 512),
#             Down(512, 512)
#         )
#
#         self.adv_layer = nn.Sequential(nn.Linear(10752, 1), nn.Sigmoid())
#
#     def forward(self, img):
#         out = self.model(img)
#         out = spatial_pyramid_pool(out, 128, [int(out.size(2)), int(out.size(3))], self.output_num)
#         out = out.view(opt.batch_size, -1)
#         validity = self.adv_layer(out)
#
#         return validity


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.output_num = [4, 2, 1]

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 1, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(2,2))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        self.adv_layer = nn.Sequential(nn.Linear(2688, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = spatial_pyramid_pool(out, 128, [int(out.size(2)), int(out.size(3))], self.output_num)
        out = out.view(opt.batch_size, -1)
        validity = self.adv_layer(out)

        return validity


if __name__ == '__main__':
    net = Discriminator()
    print(net)