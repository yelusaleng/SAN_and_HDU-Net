import argparse


def set():
    parser = argparse.ArgumentParser()

    # ----------------------------------------------
    #     Training
    # ----------------------------------------------
    parser.add_argument('--img_train', default='./')
    parser.add_argument('--mask_train', default='./')
    parser.add_argument('--edge_train', default='./')
    parser.add_argument('--img_val', default='./')
    parser.add_argument('--mask_val', default='./')
    parser.add_argument('--edge_val', default='./')
    parser.add_argument('--val_percent', default=0.1)
    parser.add_argument('--grad_accumulate', default=False)
    parser.add_argument('--is_tensorboard', default=False)
    parser.add_argument('--channel', default=3)
    parser.add_argument("--epochs", default=45, help="number of epochs of training")
    parser.add_argument("--batch_size", default=48, help="HDU-Net == 12, others == 24")
    parser.add_argument('--num_workers', default=4)
    parser.add_argument('--bilinear', default=False)
    parser.add_argument('--size', default= (192, 128))
    parser.add_argument('--multi_gpu_train', default=True)

    opt = parser.parse_args()

    return opt
