"""Evaluation entry point: full mIoU of a trained HRNet checkpoint on VOC val.

Prints the overall mIoU / pixel_acc / mean_acc AND the per-class IoU table
(worst class first).

Usage:
    python eval.py                       # simple head, outputs_simple/best.pt
    python eval.py --head ocr            # OCR head,    outputs_ocr/best.pt
    python eval.py --weights path/to.pt  # a specific checkpoint
    python eval.py --max-batches 100     # quick biased spot-check

--head MUST match how the checkpoint was trained: the two heads have different
parameter shapes, so a mismatch fails loudly at load_state_dict (good --
unlike a silent wrong-config eval). When --weights is omitted, best.pt is
taken from the folder matching --head.
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

import config
from model.hrnet import HRNet
from dataset.voc import VOCSegDataset
from utils.metrics import compute_miou
from train import get_device  # reuse the device picker


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate HRNetV2-W32 (mIoU) on VOC2012 seg val")
    p.add_argument("--head", choices=["simple", "ocr"], default=config.HEAD,
                   help="head the checkpoint was trained with (MUST match)")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in the folder "
                        "matching --head)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--max-batches", type=int, default=None,
                   help="limit val images (quick check; batch_size is 1)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    if args.weights is None:
        args.weights = os.path.join(config.output_dir_for_head(args.head), "best.pt")

    # Val set = VOC2012 seg val at original resolution (padded to /32 --
    # HRNet's stride-32 branch needs it), batch_size=1 because every image
    # keeps its own size.
    val_set = VOCSegDataset(image_set="val", train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Val images: {len(val_set)}")

    # Build model and load weights (pretrained=None: the checkpoint already
    # contains trained weights, no ImageNet file needed for eval). The head
    # knobs come from config.py -- the same source train.py used.
    model = HRNet(num_classes=config.NUM_CLASSES,
                  head=args.head,
                  pretrained=None,
                  channels=config.HRNET_CHANNELS,
                  num_blocks=config.HRNET_NUM_BLOCKS,
                  num_modules=config.HRNET_NUM_MODULES,
                  ocr_mid_channels=config.OCR_MID_CHANNELS,
                  ocr_key_channels=config.OCR_KEY_CHANNELS,
                  head_dropout=config.HEAD_DROPOUT).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights: {args.weights}")

    compute_miou(model, val_loader, device, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
