from os.path import splitext
from os import listdir
import numpy as np
import skimage.transform
import torch, os, logging
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import rotate, resize, rescale
from skimage.filters import gaussian
from skimage.util import random_noise
from PIL import Image
import matplotlib.pyplot as plt
from cv2 import GaussianBlur

from options import set

opt = set()

import albumentations as A


def strong_aug(p=0.5):
    return A.Compose([
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.8),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.4),
    ], p=p)

augmentation = strong_aug(p=0.9)


class BasicDataset_train(Dataset):
    def __init__(self, imgs_dir, masks_dir, edges_dir, scale, ):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.edges_dir = edges_dir
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img_nd, scale=None, rescale_img=None):
        img_nd = img_nd.resize((scale), resample=2)

        w, h = opt.size

        if rescale_img == True:
            img_nd = img_nd.resize((int(w / 4), int(h / 4)), resample=2)
            img_nd = np.asarray(img_nd)
        else:
            img_nd = np.asarray(img_nd)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        return img_nd

    @classmethod
    def HWC_to_CHW(cls, img_nd):
        img_trans = img_nd.transpose((2, 0, 1)).astype(np.float32)
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, index):
        idx = self.ids[index]

        img_file = self.imgs_dir + idx + '.jpg'
        mask_file = self.masks_dir + idx + '.png'
        edge_file = self.edges_dir + idx + '.png'

        img = Image.open(img_file)
        mask = Image.open(mask_file)
        edge = Image.open(edge_file)

        img = self.preprocess(img, self.scale, rescale_img=False)
        mask = self.preprocess(mask, self.scale, rescale_img=True)
        edge = self.preprocess(edge, self.scale, rescale_img=True)

        data = {"image": img, "mask": mask}
        augmented = augmentation(**data)
        img, mask = augmented["image"], augmented["mask"]

        img = self.HWC_to_CHW(img)
        mask = self.HWC_to_CHW(mask)
        edge = self.HWC_to_CHW(edge)

        return {'image': torch.from_numpy(img),
                'mask': torch.from_numpy(mask),
                'edge': torch.from_numpy(edge),
                'img_file': '{}.jpg'.format(idx)}


class BasicDataset_val(Dataset):
    def __init__(self, imgs_dir, masks_dir, edges_dir, scale):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.edges_dir = edges_dir
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img_nd, scale=None):
        if img_nd.mode != 'RGB':
            img_nd = img_nd.convert('RGB')

        pil_img = img_nd.resize((scale), resample=2)
        img_nd = np.array(pil_img)

        return img_nd

    @classmethod
    def preprocess_mask(cls, img_nd, scale=None, rescale_img=None):
        img_nd = img_nd.resize((scale), resample=2)
        w, h = opt.size

        if rescale_img == True:
            img_nd = img_nd.resize((int(w / 4), int(h / 4)), resample=2)
            img_nd = np.asarray(img_nd)
        else:
            img_nd = np.asarray(img_nd)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        return img_nd

    @classmethod
    def HWC_to_CHW(cls, img_nd):
        img_trans = img_nd.transpose((2, 0, 1)).astype(np.float32)
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, index):
        idx = self.ids[index]

        img_file = self.imgs_dir + idx + '.jpg'
        mask_file = self.masks_dir + idx + '_mask.png'
        edge_file = self.edges_dir + idx + '_edge.png'

        img = Image.open(img_file)
        mask = Image.open(mask_file)
        edge = Image.open(edge_file)

        img = self.preprocess_img(img, scale=opt.size)
        mask = self.preprocess_mask(mask, scale=opt.size, rescale_img=True)
        edge = self.preprocess_mask(edge, scale=opt.size, rescale_img=True)

        img = self.HWC_to_CHW(img)
        mask = self.HWC_to_CHW(mask)
        edge = self.HWC_to_CHW(edge)

        return {'image': torch.from_numpy(img),
                'mask': torch.from_numpy(mask),
                'edge': torch.from_numpy(edge),
                'img_file': '{}.jpg'.format(idx)}


class BasicDataset_predict(Dataset):
    def __init__(self, dataset, imgs_dir, masks_dir, edges_dir, scale, attack, param):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.edges_dir = edges_dir
        self.scale = scale
        self.attack = attack
        self.param = param
        self.dataset = dataset

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, pil_img, scale=None, img_dir=None, attack=None, param=None):
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        pil_img = pil_img.resize((scale), resample=2)
        img_nd = np.array(pil_img)

        # ----------------------------------------------
        #     opt.experiments is for `Predicting`
        # ----------------------------------------------
        assert attack in ['compression', 'noise', 'rotate', 'resize', 'blur']

        if attack == 'noise':
            img_nd = random_noise(img_nd, mode='gaussian', var=param)
        elif attack == 'blur':
            img_nd = GaussianBlur(img_nd, (param, param), 0)
        elif attack == 'rotate':
            img_nd = rotate(img_nd, param)
        elif attack == 'resize':
            img_nd = resize(img_nd, (img_nd.shape[0] * param, img_nd.shape[1] * param))
        elif attack == 'compression':
            if param == 100:
                pass
            else:
                pil_img = Image.fromarray(img_nd)
                pil_img.save('{}_compression.jpg'.format(img_dir), quality=param)
                img_nd = io.imread('{}_compression.jpg'.format(img_dir))
                os.remove('{}_compression.jpg'.format(img_dir))

        return img_nd

    @classmethod
    def preprocess_mask(cls, img_nd, scale=None, rescale_img=None, attack=None, param=None):
        img_nd = img_nd.resize((scale), resample=2)
        w, h = opt.size

        if rescale_img == True:
            img_nd = img_nd.resize((int(w / 4), int(h / 4)), resample=2)
            img_nd = np.asarray(img_nd)
        else:
            img_nd = np.asarray(img_nd)

        if attack == 'rotate':
            img_nd = rotate(img_nd, param)
        elif attack == 'resize':
            img_nd = rescale(img_nd, param)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        return img_nd

    @classmethod
    def HWC_to_CHW(cls, img_nd):
        img_trans = img_nd.transpose((2, 0, 1)).astype(np.float32)
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, index):
        idx = self.ids[index]

        img_file = self.imgs_dir + idx + '.jpg'
        if 'CASIA' in self.dataset:
            mask_file = self.masks_dir + idx + '_gt.png'
        else:
            mask_file = self.masks_dir + idx + '_mask.png'
        edge_file = self.edges_dir + idx + '_edge.png'

        img = Image.open(img_file)
        mask = Image.open(mask_file)
        edge = Image.open(edge_file)

        img = self.preprocess_img(img, scale=opt.size, img_dir=self.imgs_dir + idx, attack=self.attack, param=self.param)
        mask = self.preprocess_mask(mask, scale=opt.size, rescale_img=True, attack=self.attack,param=self.param)
        edge = self.preprocess_mask(edge, scale=opt.size, rescale_img=True, attack=self.attack, param=self.param)

        img = self.HWC_to_CHW(img)
        mask = self.HWC_to_CHW(mask)
        edge = self.HWC_to_CHW(edge)

        return {'image': torch.from_numpy(img),
                'mask': torch.from_numpy(mask),
                'edge': torch.from_numpy(edge),
                'img_file': '{}.jpg'.format(idx)}
