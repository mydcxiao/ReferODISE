import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Runs the model."
        )
    )
    parser.add_argument('--amsgrad', action='store_true',
                        help='if true, set amsgrad to True in an Adam or AdamW optimizer.')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--ck_dir', default='./checkpoints/', help='path where to save checkpoint weights')
    parser.add_argument('--ck', default='', help='path to checkpoint')
    parser.add_argument('--ck_image_encoder', default='./pretrained/image_encoder/ldm_encoder/caption_backbone.pth', help='path to checkpoint')
    parser.add_argument('--ck_prompt_encoder', default='', help='path to checkpoint')
    parser.add_argument('--ck_mask_decoder', default='', help='path to checkpoint')
    parser.add_argument('--ck_pixel_decoder', default='./pretrained/pixel_decoder/caption_pixel_decoder.pth', help='path to checkpoint')
    parser.add_argument('--dataset', default='refcoco', help='refcoco, refcoco+, or refcocog')
    parser.add_argument('--device', default='cuda:0', help='device')  # only used when testing on a single machine
    parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--img_size', default=512, type=int, help='input image size')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--log_dir', default='', help='path where to save logs')
    parser.add_argument('--lr', default=0.0001, type=float, help='the initial learning rate')
    parser.add_argument('--lr_min', default=1.0e-7, type=float, help='the minimum learning rate for cosine scheduler')
    parser.add_argument('--layer_ld', default=0.8, type=float, help='the layer-wise learning rate decay')
    parser.add_argument('--model', default='default', help='model: vit_h, vit_l, vit_b')
    parser.add_argument('--pin_mem', action='store_true',
                        help='If true, pin memory when using the data loader.')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--refer_data_root', default='./refer/data/', help='REFER dataset root directory')
    parser.add_argument('--split', default='train', help='split for dataset, e.g. train, val, or test')
    parser.add_argument('--splitBy', default='unc', help='change to umd or google when the dataset is G-Ref (RefCOCOg)')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--warmup', default=250, type=int, help='warmup epochs')
    parser.add_argument('--wd', '--weight-decay', default=0.05, type=float, metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument('--coef_dice_loss', default=1.0, type=float, help='loss weight for dice loss')
    parser.add_argument('--coef_focal_loss', default=1.0, type=float, help='loss weight for focal loss')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')

    return parser