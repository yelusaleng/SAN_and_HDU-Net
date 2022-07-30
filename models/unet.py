from blurpool import *


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.in_ch2 = in_ch + out_ch
        self.out_ch2 = self.in_ch2 * 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.in_ch2, self.out_ch2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_ch2),
            nn.ReLU(inplace=True)
        )
        self.in_ch3 = self.in_ch2 + self.out_ch2
        self.out_ch3 = self.in_ch3 * 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.in_ch3, self.out_ch3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.out_ch3),
            nn.ReLU(inplace=True)
        )
        self.in_ch4 = self.in_ch3 + self.out_ch3
        self.bottleneck = nn.Sequential(
            nn.Conv2d(self.in_ch4, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        x1 = torch.cat([out1, x], dim=1)
        out2 = self.conv2(x1)
        x2 = torch.cat([out2, x1], dim=1)
        out3 = self.conv3(x2)
        x3 = torch.cat([out3, x2], dim=1)
        out = self.bottleneck(x3)

        return out


class First_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_down, self).__init__()
        self.conv = DenseBlock(in_ch, out_ch)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)

        return x


class Down(nn.Module):
    def __init__(self, n_channels, in_ch, out_ch):
        super(Down, self).__init__()
        self.conv = DenseBlock(in_ch, out_ch)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            Downsample(channels=in_ch, filt_size=3, stride=2))
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res = nn.Sequential(
            nn.Conv2d(n_channels, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, original_input, x):
        x = self.pool(x)
        x = self.conv(x)
        x = F.relu(self.res(original_input) + x, inplace=True)

        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=None):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                nn.BatchNorm2d(in_ch // 2)
            )

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x




class First_down_fuse(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(First_down_fuse, self).__init__()
        self.conv = DenseBlock(in_ch, out_ch)

    def forward(self, x):
        x = F.relu(self.conv(x), inplace=True)

        return x


class Down_fuse(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_fuse, self).__init__()
        self.conv = DenseBlock(in_ch, out_ch)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=1),
            Downsample(channels=in_ch, filt_size=3, stride=2))
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, feats):
        x = self.pool(x)
        x = feats + x
        x = F.relu(self.conv(x), inplace=True)

        return x


# ~~~~~~~~~~ RRU-Net ~~~~~~~~~~

class RRU_double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=2, dilation=2),
            nn.GroupNorm(32, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class RRU_first_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_first_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch)
        )
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + F.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(RRU_down, self).__init__()
        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x):
        x = self.pool(x)
        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(ft1 + self.res_conv(x))
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + F.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3


class RRU_up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(RRU_up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2),
                nn.GroupNorm(32, in_ch // 2))

        self.conv = RRU_double_conv(in_ch, out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.res_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_ch))
        self.res_conv_back = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=1, bias=False))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffY, 0,
                        diffX, 0))

        x = self.relu(torch.cat([x2, x1], dim=1))

        # the first ring conv
        ft1 = self.conv(x)
        r1 = self.relu(self.res_conv(x) + ft1)
        # the second ring conv
        ft2 = self.res_conv_back(r1)
        x = torch.mul(1 + F.sigmoid(ft2), x)
        # the third ring conv
        ft3 = self.conv(x)
        r3 = self.relu(ft3 + self.res_conv(x))

        return r3