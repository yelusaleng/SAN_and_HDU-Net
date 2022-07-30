import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from skimage import io
import cv2
from PIL import Image
from SAN.options_GAN import opt

opt = opt()


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.local_net = LocalNetwork()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=opt.channel, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=opt.patch_height // 4 * opt.patch_width // 4 * 16, out_features=1024),
            nn.ReLU(),
            nn.Dropout(opt.drop_prob),
            nn.Linear(in_features=1024, out_features=opt.img_height * opt.img_width)
        )

    def forward(self, img, target):
        '''
        :param img: (b, c, h, w)
        :return: (b, c, h, w), (b,)
        '''
        batch_size = img.size(0)
        transform_img, transform_target = self.local_net(img, target)

        conv_output = self.conv(transform_img).view(batch_size, -1)
        predict = self.fc(conv_output)

        return transform_img, transform_target, predict


class LocalNetwork(nn.Module):
    def __init__(self):
        super(LocalNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_features=opt.channel * opt.patch_height * opt.patch_width,
                      out_features=20),
            nn.Tanh(),
            nn.Dropout(opt.drop_prob),
            nn.Linear(in_features=20, out_features=6),
            nn.Tanh(),
        )
        bias = torch.from_numpy(np.array([1, 0, 0, 0, 1, 0]))

        nn.init.constant_(self.fc[3].weight, 0)
        self.fc[3].bias.data.copy_(bias)


    def forward(self, img, target):
        '''

        :param img: (b, c, h, w)
        :return: (b, c, h, w)
        '''
        batch_size = img.size(0)
        theta = self.fc(img.view(batch_size, -1)).view(batch_size, 2, 3)
        grid = F.affine_grid(theta, img.shape)

        # transform image
        img_transform = F.grid_sample(img, grid)
        target_transform = F.grid_sample(target, grid)

        return img_transform, target_transform


def extract_rect(img, mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    img_rect = img[y:y + h, x:x + w].astype(np.float32)
    mask_rect = mask[y:y + h, x:x + w].astype(np.float32)

    # img_rect = resize(img_rect, [opt.patch_height, opt.patch_width])
    # mask_rect = resize(mask_rect, [opt.patch_height, opt.patch_width])

    return img_rect, mask_rect


def insert_patch(image, patch, patch_mask, placement):
    assert type(placement) == tuple

    insert_patch = Image.fromarray(patch)
    insert_patch_mask = Image.fromarray(patch_mask).convert('L')

    image.paste(insert_patch, placement, insert_patch_mask)

    return np.array(image)



if __name__ == '__main__':
    tam = '2472627d9b38bce396254ac17b9b3655.jpg'
    mask = '2472627d9b38bce396254ac17b9b3655_mask.png'

    img_tam = io.imread(tam)
    img_mask = io.imread(mask)
    img_mask = cv2.merge([img_mask, img_mask, img_mask]).astype(np.uint8)
    forg = cv2.bitwise_and(img_tam, img_mask)

    forg, forg_mask = extract_rect(forg, img_mask)

    # forg = np.array(forg).transpose((2, 0, 1)).astype(np.float32)
    # forg = torch.tensor(forg).unsqueeze(0)

    # forg_mask = np.array(forg_mask).transpose((2, 0, 1)).astype(np.float32)
    # forg_mask = torch.tensor(forg_mask).unsqueeze(0)

    # net = Network()
    # patch_transform, target_transform, predict = net(forg, forg_mask)

    # patch_transform = tensor_to_np(patch_transform)
    # target_transform = tensor_to_np(target_transform)

    # placement = predict.argmax().numpy()
    # placement = (placement // 384, placement % 384)

    # image = Image.open('8569531f0cfe6fed6f0911100c8c8d56_hor.jpg')
    # image_pasted = insert_patch(image, patch_transform, target_transform, placement)

    # k = cv2.bitwise_and(img_tam[0:opt.patch_width, 0:opt.patch_height], forg_mask)


