import argparse

def opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_path_img', type=str, default='/home/hdd_2T/coco_forgery/SAN_coco_10w/img/')
    parser.add_argument('--save_path_mask', type=str, default='/home/hdd_2T/coco_forgery/SAN_coco_10w/mask/')
    # parser.add_argument('--img_train2', type=str, default='/home/hdd_2T/HDU-Net/SAN/data/original/tam/')
    # parser.add_argument('--mask_train2', type=str, default='/home/hdd_2T/HDU-Net/SAN/data/original/mask/')
    parser.add_argument('--img_train2', type=str, default='/home/hdd_2T/coco_forgery/train2017/')
    parser.add_argument('--mask_train2', type=str, default='/home/hdd_2T/coco_forgery/train2017_binaryMask/')
    # parser.add_argument('--img_train2', type=str, default='C:/Users\Administrator\Desktop\Datasets\COCO\splicing\img/')
    # parser.add_argument('--mask_train2', type=str, default='C:/Users\Administrator\Desktop\Datasets\COCO\splicing\mask/')
    # parser.add_argument('--img_height', type=int, default=256)
    # parser.add_argument('--img_width', type=int, default=384)
    # parser.add_argument('--patch_height', type=int, default=100)
    # parser.add_argument('--patch_width', type=int, default=100)
    # parser.add_argument('--channel', type=int, default=3)
    # parser.add_argument('--drop_prob', type=float, default=0.8)
    parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--lr", type=float, default=3*1e-2, help="adam: learning rate")
    # parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    # parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    # parser.add_argument("--sample_interval", type=int, default=2, help="interval between image sampling")

    opt = parser.parse_args()

    return opt
