from os.path import splitext
from os import listdir
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
import logging, cv2, os
from PIL import Image
from skimage.morphology import convex_hull_image
from skimage import io, filters

from options_GAN import opt

opt = opt()


class Dataset_forTrain(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img = np.array(pil_img)
        if img.ndim == 3 and img.shape[2] != 3:
            img = img[:, :, :3]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        if img.shape[0:2] != (256, 256):
            img = resize(img, (256, 256)) * 255
            img = np.uint8(img)

        return img

    @classmethod
    def HWC_to_CHW(cls, img):
        img_trans = img.transpose((2, 0, 1))

        return img_trans

    def __getitem__(self, index):
        idx = self.ids[index]

        mask_file = self.masks_dir + idx + '.png'
        img_file = self.imgs_dir + idx + '.jpg'

        mask = Image.open(mask_file)
        img = Image.open(img_file)

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        image = cv2.imread(img_file)
        img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_clarity = cv2.Laplacian(img2gray, cv2.CV_64F).var()

        edge = filters.sobel(img[:,:,0]) * 255
        edge = np.asarray(edge, np.uint8)
        _, edge = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)
        img_complex = len(np.nonzero(edge)[0]) / (img.shape[0] * img.shape[1])

        img = self.HWC_to_CHW(img)
        mask = self.HWC_to_CHW(mask)

        return {'image': torch.from_numpy(img),
                'mask': torch.from_numpy(mask),
                'clarity': torch.from_numpy(np.array(img_clarity)),
                'complexity': torch.from_numpy(np.array(img_complex))}


import albumentations as A
transform_generate = A.Compose([
    # A.PadIfNeeded(min_height=512, min_width=512, border_mode=0),
    # A.CropNonEmptyMaskIfExists(height=512, width=512),
    A.Resize(height=512, width=512),
])


class Dataset_forPredict(Dataset):
    def __init__(self, imgs_dir, masks_dir):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img):
        if img.ndim == 3 and img.shape[2] != 3:
            img = img[:, :, :3]
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        return img

    @classmethod
    def HWC_to_CHW(cls, img):
        img_trans = img.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, index):
        idx = self.ids[index]

        mask_file = self.masks_dir + idx + '.png'
        img_file = self.imgs_dir + idx + '.jpg'

        img = cv2.imread(img_file)
        img = img[:,:,::-1]
        mask = cv2.imread(mask_file, 0)

        data = {"image": img, "mask": mask}
        transformed = transform_generate(**data)
        img, mask = transformed["image"], transformed["mask"]

        img = self.preprocess(img)
        mask = self.preprocess(mask)

        img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_clarity = cv2.Laplacian(img2gray, cv2.CV_64F).var()

        edge = filters.sobel(img[:,:,0]) * 255
        edge = np.asarray(edge, np.uint8)
        _, edge = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)
        img_complex = len(np.nonzero(edge)[0]) / (img.shape[0] * img.shape[1])

        img = self.HWC_to_CHW(img)
        mask = self.HWC_to_CHW(mask)

        return {'image': torch.from_numpy(img.copy()),
                'mask': torch.from_numpy(mask.copy()),
                'clarity': torch.from_numpy(np.array(img_clarity)),
                'complexity': torch.from_numpy(np.array(img_complex))
                }


def extract_single_instance(mask):
    # use conv hell to fill the tiny hole of mask (due to the flaws of original masks)
    chull = convex_hull_image(mask.squeeze().astype(np.bool))
    chull = chull.astype(np.uint8) * 255
    chull = cv2.merge([chull, chull, chull])

    gray = cv2.cvtColor(chull, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    return [x, y, w, h]


def extract_all_instances(img_dir, mask_dir):
    # extract the coordinates of all instance to dict,
    # only restore a dict for the coords of instance since img and mask have common coords

    instances_dict = {}
    lightInf_dict = {}
    clarity_dict = {}
    complexity_dict = {}

    for index, i in enumerate(tqdm(os.listdir(img_dir))):
        img = cv2.imread(os.path.join(img_dir, i))
        img = img[:,:,::-1]
        img_name = i.split('.')[0]
        mask_path = os.path.join(mask_dir, '{}.png'.format(img_name))
        mask = cv2.imread(mask_path)
        [x, y, w, h] = extract_single_instance(mask[:,:,0])

        inserted_region = cv2.bitwise_and(img, mask)
        inserted_region = inserted_region[y:y + h, x:x + w]
        instances_dict['{}'.format(img_name)] = [x, y, w, h]
        # ----------------------------------------------
        #     about light information
        # ----------------------------------------------
        lightInf_dict['{}'.format(img_name)] = np.mean(inserted_region[np.nonzero(inserted_region)])
        # ----------------------------------------------
        #     about image clarity
        # ----------------------------------------------
        img2gray = cv2.cvtColor(inserted_region, cv2.COLOR_BGR2GRAY)
        imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
        clarity_dict['{}'.format(img_name)] = imageVar
        # ----------------------------------------------
        #     about image complexity
        # ----------------------------------------------
        edge = filters.sobel(inserted_region[:,:,0]) * 255
        edge = np.asarray(edge, np.uint8)
        _, edge = cv2.threshold(edge, 30, 255, cv2.THRESH_BINARY)
        img_complex = len(np.nonzero(edge)[0]) / (inserted_region.shape[0] * inserted_region.shape[1])
        complexity_dict['{}'.format(img_name)] = img_complex

    return instances_dict, lightInf_dict, clarity_dict, complexity_dict


def dict2matrix(instances_dict, key, path_img, path_mask):
    # map the coordinates of instance to matrix
    [x, y, w, h] = instances_dict[key]

    img = os.path.join(path_img, '{}.jpg'.format(key))
    mask = os.path.join(path_mask, '{}.png'.format(key))
    img = cv2.imread(img)
    img = img[:,:,::-1]
    mask = cv2.imread(mask)
    foreground = cv2.bitwise_and(img, mask)

    instance = foreground[y:y + h, x:x + w]
    instance_mask = mask[y:y + h, x:x + w]

    # instance = cv2.resize(instance, (opt.patch_height, opt.patch_width), cv2.INTER_LINEAR)
    # instance_mask = cv2.resize(instance_mask, (opt.patch_height, opt.patch_width), cv2.INTER_LINEAR)

    # instance, instance_mask = torch.from_numpy(HWC_to_CHW(instance)), torch.from_numpy(HWC_to_CHW(instance_mask))

    return instance, instance_mask


if __name__ == '__main__':
    print('1')