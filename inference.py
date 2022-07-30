import logging
import os
import time
import cv2

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage import io
from pytorch_colors import rgb_to_hed, rgb_to_lab, rgb_to_yuv

from models.networks import *
from utils.dataset import BasicDataset_predict
from utils.sobel import Sobel
from utils.srm import SRM

from options import set

opt = set()

import warnings

warnings.filterwarnings('ignore')


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return (mask * 255).astype(np.uint8)


def truncate_2(x):
    neg = ((x + 2) + abs(x + 2)) / 2 - 2
    return -(-neg + 2 + abs(- neg + 2)) / 2 + 2


def color_spaces(imgs):
    assert imgs.dim() == 4

    SRM_kernel = SRM()
    imgs_srm = SRM_kernel(imgs)
    imgs_srm = truncate_2(imgs_srm)
    imgs_a = rgb_to_lab(imgs)[:, 1, :, :].unsqueeze(1)
    imgs_v = rgb_to_yuv(imgs)[:, 2, :, :].unsqueeze(1)

    return list([imgs, imgs_srm, imgs_a, imgs_v])


def sobel_operate(img):
    if opt.use_sobel:
        sobel = Sobel(stride=1, channels=3)
        img_sobel = sobel(img)
        img = img_sobel + img
        return img


if __name__ == "__main__":
    model = 'best_model.pth'

    print("Loading model {}".format(model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device {}'.format(device))

    if opt.multi_gpu_train:
        kwargs = {'map_location': lambda storage, loc: storage.cuda(0)}
        net = Our()
        net = nn.DataParallel(net)
        net.load_state_dict(torch.load(model, **kwargs))
        net.to(device=device)
    else:
        net = Our()
        net.load_state_dict(torch.load(model))
        net.to(device=device)
        torch.backends.cudnn.benachmark = True

    print('"Model loaded !"')
    time.sleep(0.5)

    img = 'test.jpg'
    name = img.split('/')[1].split('.')[0]
    pil_img = Image.open(img)
    h, w = np.asarray(pil_img).shape[0], np.asarray(pil_img).shape[1]

    img = torch.from_numpy(
        BasicDataset_predict.HWC_to_CHW(
            BasicDataset_predict.preprocess_img(pil_img, scale=opt.size, attack='compression', param=100)
        ))
    img = img.unsqueeze(0)
    img = img.to(torch.float32)
    img = img.cuda(non_blocking=True)

    net.eval()
    with torch.no_grad():
        if img.dim() == 4:
            img = img[:, :3, :, :]
        assert img.dim() == 4

        img = color_spaces(img)
        pred, _, _, _, _ = net(img)
        prob = torch.sigmoid(pred)

        full_mask = prob.squeeze().cpu().numpy()
        full_mask = resize(full_mask, (h, w))

    plt.imsave(os.path.join('Heatmap.png'), full_mask, format='png',cmap='jet')

    result = mask_to_image(full_mask > 0.5)
    _, bina = cv2.threshold(result, 127, 255, cv2.THRESH_BINARY)
    bina = cv2.merge([bina, bina, bina, bina])

    # plt.subplot(121)
    # plt.imshow(np.array(pil_img))
    # plt.subplot(122)
    # plt.imshow(full_mask)
    # plt.show()