"""Evaluation entry point: full mIoU of a trained checkpoint on VOC2012 val.

Prints the overall mIoU / pixel_acc / mean_acc AND the per-class IoU table
(worst class first). Note there is no separate eval_per_class.py like the
FCOS project needed: segmentation's per-class numbers fall out of the same
confusion matrix for free -- one script does both jobs.

Usage:
    python eval.py                          # uses outputs/best.pt
    python eval.py --weights outputs/last.pt --device cpu
    python eval.py --max-batches 100        # quick biased spot-check
"""

import argparse

import torch
from torch.utils.data import DataLoader

import config
from model.deeplab import DeepLab
from dataset.voc import VOCSegDataset
from utils.metrics import compute_miou
from train import get_device  # reuse the device picker


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DeepLab (mIoU) on VOC2012 seg val")
    p.add_argument("--weights", default=f"{config.OUTPUT_DIR}/best.pt", help="checkpoint path")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--max-batches", type=int, default=None,
                   help="limit val images (quick check; batch_size is 1)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Val set = VOC2012 seg val at original resolution (padded to /8),
    # batch_size=1 because every image keeps its own size.
    val_set = VOCSegDataset(image_set="val", train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Val images: {len(val_set)}")

    # Build model and load weights (pretrained=False: the checkpoint already
    # contains trained weights, no need to fetch ImageNet ones first).
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=False, backbone=config.BACKBONE,
                    neck_hidden_channels=config.NECK_HIDDEN_CHANNELS,
                    neck_out_channels=config.NECK_OUT_CHANNELS,
                    atrous_rate=config.ATROUS_RATE,
                    neck_dropout=config.NECK_DROPOUT).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights: {args.weights}")

    compute_miou(model, val_loader, device, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
