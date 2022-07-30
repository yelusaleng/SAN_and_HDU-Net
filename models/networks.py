from models.unet import *
from options import set
from kornia.geometry.transform import PyrDown
from skimage import io
import matplotlib.pyplot as plt

opt = set()


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class sub_Our(nn.Module):
    def __init__(self, n_channels=4, n_classes=1, bilinear=False):
        super(sub_Our, self).__init__()

        self.kernel_num = 24

        self.down = First_down(n_channels, self.kernel_num)
        self.down1 = Down(n_channels, self.kernel_num, self.kernel_num)
        self.down2 = Down(n_channels, self.kernel_num, self.kernel_num)
        self.down3 = Down(n_channels, self.kernel_num, self.kernel_num)
        self.down4 = Down(n_channels, self.kernel_num, self.kernel_num)
        self.up1 = Up(self.kernel_num * 2, self.kernel_num, bilinear=bilinear)
        self.up2 = Up(self.kernel_num * 2, self.kernel_num, bilinear=bilinear)
        self.out = OutConv(self.kernel_num, n_classes)

    def forward(self, x):
        x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x2 = F.interpolate(x1, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x2, scale_factor=0.5, mode='bilinear')
        x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear')

        dx1 = self.down(x)
        dx2 = self.down1(x1, dx1)
        dx3 = self.down2(x2, dx2)
        dx4 = self.down3(x3, dx3)
        dx5 = self.down4(x4, dx4)

        ux1 = self.up1(dx5, dx4)
        ux2 = self.up2(ux1, dx3)
        ux3 = self.out(ux2)

        return dx2, dx3, dx4, dx5, ux3


class Our(nn.Module):
    def __init__(self, n_classes=1, bilinear=False):
        super(Our, self).__init__()

        self.kernel_num = 24

        self.subnet_1 = sub_Our(3)
        self.subnet_2 = sub_Our(3)
        self.subnet_3 = sub_Our(1)
        self.subnet_4 = sub_Our(1)

        self.down = First_down_fuse(8, self.kernel_num)
        self.down1 = Down_fuse(self.kernel_num, self.kernel_num)
        self.down2 = Down_fuse(self.kernel_num, self.kernel_num)
        self.down3 = Down_fuse(self.kernel_num, self.kernel_num)
        self.down4 = Down_fuse(self.kernel_num, self.kernel_num)
        self.up1 = Up(self.kernel_num * 2, self.kernel_num, bilinear=bilinear)
        self.up2 = Up(self.kernel_num * 2, self.kernel_num, bilinear=bilinear)
        self.out = OutConv(self.kernel_num, n_classes)

    def forward(self, x):
        dx2_1, dx3_1, dx4_1, dx5_1, ux3_1 = self.subnet_1(x[0])
        dx2_2, dx3_2, dx4_2, dx5_2, ux3_2 = self.subnet_2(x[1])
        dx2_3, dx3_3, dx4_3, dx5_3, ux3_3 = self.subnet_3(x[2])
        dx2_4, dx3_4, dx4_4, dx5_4, ux3_4 = self.subnet_4(x[3])

        input = torch.cat((x[0], x[1], x[2], x[3]), dim=1)

        dx1_fuse = self.down(input)
        dx2_fuse = self.down1(dx1_fuse, dx2_1 + dx2_2 + dx2_3 + dx2_4)
        dx3_fuse = self.down2(dx2_fuse, dx3_1 + dx3_2 + dx3_3 + dx3_4)
        dx4_fuse = self.down3(dx3_fuse, dx4_1 + dx4_2 + dx4_3 + dx4_4)
        dx5_fuse = self.down4(dx4_fuse, dx5_1 + dx5_2 + dx5_3 + dx5_4)

        ux1_fuse = self.up1(dx5_fuse, dx4_fuse)
        ux2_fuse = self.up2(ux1_fuse, dx3_fuse)
        ux3_fuse = self.out(ux2_fuse)

        return ux3_fuse, ux3_1, ux3_2, ux3_3, ux3_4
