import time, logging, os
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.parallel

from pytorch_colors import rgb_to_hed, rgb_to_lab, rgb_to_yuv

from models.networks import *
from utils.sobel import Sobel
from utils.srm import SRM
from dice_loss import dice_coeff
from utils.dataset import BasicDataset_train, BasicDataset_val

from options import set

opt = set()

import warnings

warnings.filterwarnings('ignore')


def truncate_2(x):
    neg = ((x + 2) + abs(x + 2)) / 2 - 2
    return -(-neg + 2 + abs(- neg + 2)) / 2 + 2


def space_transfer(imgs):
    assert imgs.dim() == 4

    # imgs_hed = rgb_to_hed(imgs)
    SRM_kernel = SRM()
    imgs_srm = SRM_kernel(imgs)
    imgs_srm = truncate_2(imgs_srm)
    imgs_a = rgb_to_lab(imgs)[:, 1, :, :].unsqueeze(1)
    imgs_v = rgb_to_yuv(imgs)[:, 2, :, :].unsqueeze(1)

    return list([imgs, imgs_srm, imgs_a, imgs_v])


def sobel_operation(img, mask):
    sobel = Sobel(stride=1, channels=3)
    img_sobel = sobel(img)
    img = 0.5 * img_sobel + 0.5 * img
    return img, mask


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return lr


def train_net(net,
              device,
              lr=1e-2,
              save_cp=True):
    if opt.is_tensorboard:
        writer = SummaryWriter(comment=f'LR_{lr}_BS_{opt.batch_size}')
    global_step = 0

    net = torch.nn.DataParallel(net).cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0)
    criterion = nn.BCEWithLogitsLoss()

    all_val_scores = []

    for epoch in range(opt.epochs):
        if epoch >= 15 and epoch < 30:
            set_learning_rate(optimizer, 1e-3)
        elif epoch >= 30:
            set_learning_rate(optimizer, 1e-4)

        train = BasicDataset_train(opt.img_train, opt.mask_train, opt.edge_train, opt.size)
        val = BasicDataset_val(opt.img_val, opt.mask_val, opt.edge_val, opt.size)
        n_train = len(train)
        n_val = len(val)

        train_loader = DataLoader(train, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=opt.num_workers, pin_memory=True)
        val_loader = DataLoader(val, batch_size=opt.batch_size, shuffle=False,
                                num_workers=opt.num_workers, pin_memory=True, drop_last=True)

        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{opt.epochs}', unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                imgs = batch['image']
                true_masks = batch['mask']
                edges = batch['edge']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                edges = edges.to(device=device, dtype=torch.float32)

                imgs = space_transfer(imgs)
                net_out, out_1, out_2, out_3, out_4 = net(imgs)
                loss_out = criterion(net_out, true_masks)
                loss_1 = criterion(out_1, true_masks)
                loss_2 = criterion(out_2, true_masks)
                loss_3 = criterion(out_3, edges)
                loss_4 = criterion(out_4, edges)
                loss = (loss_out + loss_1 + loss_2 + loss_3 + loss_4) / 5
                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if opt.grad_accumulate:
                    # gradient accumulation
                    accumulation_steps = 4
                    loss = loss / accumulation_steps
                    loss.backward()
                    if ((i + 1) % accumulation_steps) == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                elif not opt.grad_accumulate:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                pbar.update(opt.batch_size)

        net.eval()

        tot = 0
        with torch.no_grad():
            for j, batch in enumerate(val_loader):
                val_imgs = batch['image']
                val_true_masks = batch['mask']
                val_imgs = val_imgs.to(device=device, dtype=torch.float32)
                val_true_masks = val_true_masks.to(device=device, dtype=torch.float32)

                val_imgs = space_transfer(val_imgs)
                val_pred, _, _, _, _ = net(val_imgs)
                pred = (F.sigmoid(val_pred) > 0.5).float()
                tot += dice_coeff(pred, val_true_masks).item()

        time.sleep(0.5)

        print('Learning rate: {}'.format(get_learning_rate(optimizer)[0]))
        print('Training Loss: {:.4f}'.format(epoch_loss / (n_train / opt.batch_size)))
        print('Dice Coeff: {:.4f}'.format(tot / (n_val / opt.batch_size)))

        all_val_scores.append(tot / (n_val / opt.batch_size))

        # save checkpoint
        if save_cp:
            if not os.path.exists(dir_checkpoint):
                os.makedirs(dir_checkpoint)
                logging.info('Created checkpoint directory')
            torch.save(net.state_dict(),
                       dir_checkpoint + 'Epoch{}_LS-{:.3f}_DC-{:.3f}.pth'.format(epoch + 1, epoch_loss / (n_train / opt.batch_size), tot / (n_val / opt.batch_size)))
            print('Checkpoint {} saved !'.format(epoch + 1))
            print()
            time.sleep(0.5)

        # save best model parameters
        curr_model = all_val_scores[-1]
        if curr_model >= max(all_val_scores):
            torch.save(net.state_dict(), dir_checkpoint + 'best_model.pth')

        # draws training process
        global_step += 1

        if opt.is_tensorboard:
            writer.add_scalar('Loss/train', epoch_loss / (i + 1), global_step)
            writer.add_scalar('Dice/test', tot / (j + 1), global_step)
            writer.add_images('images/RGB', val_imgs[0], global_step)
            writer.add_images('images/SRM', val_imgs[1], global_step)
            writer.add_images('images/lab-a', val_imgs[2], global_step)
            writer.add_images('images/yuv-v', val_imgs[3], global_step)
            writer.add_images('masks/true', val_true_masks, global_step)
            writer.add_images('masks/pred', pred, global_step)

    if opt.is_tensorboard:
        writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = Our()
    net.to(device=device)
    torch.backends.cudnn.benchmark = True

    dir_checkpoint = '/home/hdd_2T/HDU-Net/logs/HU-Net2/'
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    train_net(net=net, device=device)
