import random, os, time

import cv2
import torch
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, IAAPerspective, CLAHE, RandomRotate90, JpegCompression,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

from torch.utils.data import DataLoader
from dataset import *
from models import *

from options_GAN import opt
opt = opt()

cuda = True if torch.cuda.is_available() else False


def augs_for_insert_object(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)


def HWC_to_CHW(img):
    img_trans = img.transpose((2, 0, 1))
    return torch.from_numpy(img_trans)


def select_proximate_patch(img_Inf, Inf_dict):
    tm = {}
    for key, value in Inf_dict.items():
        t = abs(img_Inf - value)
        tm[key] = t
    tm = sorted(tm.items(), key=lambda item: item[1])
    return tm[:50]


if __name__ == '__main__':
    predictor = Predictor().cuda()
    model = 'best_model.pth'
    predictor.load_state_dict(torch.load(model))

    all_instances, lightInf_dict, clarity_dict, complexity_dict = extract_all_instances(opt.img_train2, opt.mask_train2)
    time.sleep(1)

    # train for forgery datasets (use to learn "weakest" position of an img)
    data = Dataset_forPredict(opt.img_train2, opt.mask_train2)
    data_loader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    num_generate_img = 1

    pbar = tqdm(data_loader)
    for i, batch in enumerate(pbar):
        real_imgs = batch['image']
        real_masks = batch['mask']
        real_img_clarity = batch['clarity']
        real_img_complexity = batch['complexity']

        # Configure input
        real_imgs = real_imgs.cuda(non_blocking=True)
        real_masks = real_masks.cuda(non_blocking=True)
        real_img_clarity = real_img_clarity.cuda(non_blocking=True)
        real_img_complexity = real_img_complexity.cuda(non_blocking=True)
        synth_imgs = real_imgs.clone()
        synth_masks = torch.zeros_like(real_masks)

        _, prob = predictor(real_imgs.to(torch.float32) / 255)

        # -----------------
        #  Data Synthesis
        # -----------------
        augmentation = augs_for_insert_object(p=0.9)

        for m in range(opt.batch_size):
            # according to the light information of an patch, select an proximate patch for real_img
            t = [1, 2, 3]
            t = random.choice(t)
            # ----------------------------------------------
            #     about light information
            # ----------------------------------------------
            if t == 1:
                lightInf = real_imgs[m, ...].to(torch.float32).mean().item()
                keys = select_proximate_patch(lightInf, lightInf_dict)
            # ----------------------------------------------
            #     about image clarity
            # ----------------------------------------------
            elif t == 2:
                clarity = real_img_clarity[m].item()
                keys = select_proximate_patch(clarity, clarity_dict)
            # ----------------------------------------------
            #     about image complexity
            # ----------------------------------------------
            elif t == 3:
                complexity = real_img_complexity[m].item()
                keys = select_proximate_patch(complexity, complexity_dict)

            key = random.choice(keys)[0]
            instance, instance_mask = dict2matrix(all_instances, key, opt.img_train2, opt.mask_train2)

            data = {"image": instance, "mask": instance_mask}
            augmented = augmentation(**data)
            instance, instance_mask = augmented["image"], augmented["mask"],

            instance = HWC_to_CHW(instance).cuda(non_blocking=True)
            instance_mask = HWC_to_CHW(instance_mask).cuda(non_blocking=True)

            placement = prob[m, ...].squeeze().argmax()
            h_t, w_t = real_imgs.size()[3], real_imgs.size()[2]
            place_y, place_x = placement // h_t, placement % h_t

            w_margin = h_t if instance.shape[2] + place_x > h_t else instance.shape[2] + place_x
            h_margin = w_t if instance.shape[1] + place_y > w_t else instance.shape[1] + place_y

            is_inserted = real_imgs[m, :, place_y:h_margin, place_x:w_margin]
            is_inserted_mask = real_masks[m, :, place_y:h_margin, place_x:w_margin]

            insert_instance = is_inserted & ~instance_mask[:, :h_margin - place_y, :w_margin - place_x]
            insert_instance = insert_instance | instance[:, :h_margin - place_y, :w_margin - place_x]
            insert_instance_mask = is_inserted_mask | instance_mask[:, :h_margin - place_y, :w_margin - place_x]

            synth_imgs[m, :, place_y:h_margin, place_x:w_margin] = insert_instance
            synth_masks[m, :, place_y:h_margin, place_x:w_margin] = insert_instance_mask[0, :, :]

        for i in range(opt.batch_size):
            img = synth_imgs[i, :, :, :].cpu().numpy()
            img = img.transpose((1, 2, 0))[:,:,::-1]
            mask = synth_masks[i, :, :, :].cpu().numpy()
            mask = mask.transpose((1, 2, 0))

            cv2.imwrite(os.path.join(opt.save_path_img, '0000_{}.jpg'.format(num_generate_img)), img)
            cv2.imwrite(os.path.join(opt.save_path_mask, '0000_{}.png'.format(num_generate_img)), mask)

            num_generate_img += 1

        if num_generate_img > 120000:
            break
