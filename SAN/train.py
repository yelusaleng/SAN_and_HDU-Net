import random, os, time
from tqdm import tqdm

from albumentations import (
    HorizontalFlip, IAAPerspective, CLAHE, RandomRotate90, JpegCompression,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)

from torch.utils.data import DataLoader

from SAN.dataset import *
from SAN.models import *

from warnings import filterwarnings
filterwarnings('ignore')

from SAN.options_GAN import opt
opt = opt()

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def strong_aug(p=0.5):
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
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    predictor = Predictor()
    discriminator = Discriminator()

    if cuda:
        predictor.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # train for forgery datasets (use to learn "weakest" position of an img)
    train = Dataset_forTrain(opt.img_train1, opt.mask_train1, scale=1)
    train_loader = DataLoader(train, batch_size=opt.batch_size, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)

    # Optimizers
    optimizer_P = torch.optim.SGD(predictor.parameters(), lr=opt.lr, momentum=0.95, weight_decay=0.0005)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=opt.lr, momentum=0.95, weight_decay=0.0005)

    # ----------
    #  Training
    # ----------
    num_generate_img = 1

    all_instances, lightInf_dict, clarity_dict, complexity_dict = extract_all_instances(opt.img_train2, opt.mask_train2)
    time.sleep(1)

    for epoch in range(opt.n_epochs):
        # Adversarial ground truths
        valid = torch.ones((opt.batch_size, 1), requires_grad=False)
        valid = valid.to(torch.float32)
        valid = valid.cuda(non_blocking=True)
        fake = torch.zeros((opt.batch_size, 1), requires_grad=False)
        fake = fake.to(torch.float32)
        fake = fake.cuda(non_blocking=True)

        pbar = tqdm(train_loader)
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
            synth_imgs, synth_masks = real_imgs.clone(), real_masks.clone()

            _, prob = predictor(real_imgs.to(torch.float32) / 255)

            # -----------------
            #  Data Synthesis
            # -----------------
            augmentation = strong_aug(p=0.9)

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

                if instance.shape[2] + place_x > h_t:
                    w_margin = h_t
                else:
                    w_margin = instance.shape[2] + place_x

                if instance.shape[1] + place_y > w_t:
                    h_margin = w_t
                else:
                    h_margin = instance.shape[1] + place_y

                is_inserted = real_imgs[m, :, place_y:h_margin, place_x:w_margin]
                is_inserted_mask = real_masks[m, :, place_y:h_margin, place_x:w_margin]

                insert_instance = is_inserted & ~instance_mask[:, :h_margin - place_y, :w_margin - place_x]
                insert_instance = insert_instance | instance[:, :h_margin - place_y, :w_margin - place_x]
                insert_instance_mask = is_inserted_mask | instance_mask[:, :h_margin - place_y, :w_margin - place_x]

                synth_imgs[m, :, place_y:h_margin, place_x:w_margin] = insert_instance
                synth_masks[m, :, place_y:h_margin, place_x:w_margin] = insert_instance_mask[0, :, :]

            real_imgs = real_imgs.to(torch.float32) / 255
            synth_imgs = synth_imgs.to(torch.float32) / 255

            # -----------------
            #  Train Predictor
            # -----------------
            optimizer_P.zero_grad()
            p_fake_loss = adversarial_loss(discriminator(synth_imgs), valid)
            p_real_loss = adversarial_loss(prob, real_masks.to(torch.float32) / 255)
            p_loss = (p_fake_loss + p_real_loss) / 2
            p_loss.backward()
            optimizer_P.step()


            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            d_real_loss = adversarial_loss(discriminator(real_imgs), valid)
            d_fake_loss = adversarial_loss(discriminator(synth_imgs), fake)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            pbar.set_description('Epoch %d/%d' % (epoch, opt.n_epochs))
            pbar.set_postfix(P_loss=p_loss.item(), D_loss=d_loss.item())

            # if epoch >= 10:
            #     synth_imgs *= 255
            #     synth_imgs = synth_imgs.to(torch.uint8)
            #
            #     for i in range(opt.batch_size):
            #         img = synth_imgs[i, :, :, :].cpu().numpy()
            #         img = img.transpose((1, 2, 0))
            #         io.imsave(os.path.join(opt.save_path_img, '{}.jpg'.format(num_generate_img)), img, quality=100)
            #
            #         mask = synth_masks[i, :, :, :].cpu().numpy()
            #         mask = mask.transpose((1, 2, 0))
            #         io.imsave(os.path.join(opt.save_path_mask, '{}_mask.png'.format(num_generate_img)), mask)
            #
            #         num_generate_img += 1

        # torch.save(predictor.state_dict(), 'best_model.pth')

        pbar.close()

        # if num_generate_img >= 30000:
        #     break

        # save_image(gen_imgs.data, "./../../generate/images/epoch-{}.png".format(epoch), nrow=2, normalize=True)
