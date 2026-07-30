"""Evaluation entry point: full mIoU of a trained DeepLab-v3+ checkpoint.

Prints the overall mIoU / pixel_acc / mean_acc AND the per-class IoU table
(worst class first). Segmentation's per-class numbers fall out of the same
confusion matrix for free -- one script does both jobs.

Usage:
    python eval.py                          # uses outputsv3plus/best.pt
    python eval.py --weights path/to.pt     # a specific checkpoint
    python eval.py --device cpu
    python eval.py --max-batches 100        # quick biased spot-check
"""

import os
import argparse

import torch
from torch.utils.data import DataLoader

import config
from model.deeplab import DeepLab
from dataset.voc import VOCSegDataset
from utils.metrics import compute_miou
from train import get_device  # reuse the device picker


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DeepLab-v3+ (mIoU) on VOC2012 seg val")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: outputsv3plus/best.pt)")
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
        args.weights = os.path.join(config.OUTPUT_DIR_V3PLUS, "best.pt")

    # Val set = VOC2012 seg val at original resolution (padded to /8),
    # batch_size=1 because every image keeps its own size.
    val_set = VOCSegDataset(image_set="val", train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Val images: {len(val_set)}")

    # Build model and load weights (pretrained=False: the checkpoint already
    # contains trained weights, no need to fetch ImageNet ones first). The
    # model config MUST match how the checkpoint was trained -- these come from
    # config.py, the same source trainV3.py used.
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=False, backbone=config.BACKBONE,
                    aspp_rates=config.ASPP_V3_RATES,
                    aspp_hidden=config.ASPP_V3_HIDDEN,
                    low_level_proj=config.LOW_LEVEL_PROJ,
                    dropout=config.NECK_DROPOUT).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights: {args.weights}")

    compute_miou(model, val_loader, device, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
