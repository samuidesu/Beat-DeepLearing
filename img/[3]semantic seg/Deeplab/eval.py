"""Evaluation entry point: full mIoU of a trained checkpoint on VOC2012 val.

Prints the overall mIoU / pixel_acc / mean_acc AND the per-class IoU table
(worst class first). Note there is no separate eval_per_class.py like the
FCOS project needed: segmentation's per-class numbers fall out of the same
confusion matrix for free -- one script does both jobs.

Usage:
    python eval.py                          # v1 (LargeFOV), uses outputs/best.pt
    python eval.py --neck aspp              # v2 (ASPP),    uses outputsv2/best.pt
    python eval.py --weights outputs/last.pt --device cpu
    python eval.py --max-batches 100        # quick biased spot-check

The --neck flag MUST match how the checkpoint was trained (train.py -> largefov,
trainv2.py -> aspp): the neck architectures differ, so a mismatch fails at
load_state_dict. When --weights is omitted it defaults to best.pt in the folder
matching --neck (outputs/ for largefov, outputsv2/ for aspp).
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
    p = argparse.ArgumentParser(description="Evaluate DeepLab (mIoU) on VOC2012 seg val")
    p.add_argument("--neck", choices=["largefov", "aspp"], default="largefov",
                   help="neck the checkpoint was trained with: largefov (v1, "
                        "train.py) or aspp (v2, trainv2.py). MUST match.")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in the folder "
                        "matching --neck)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--max-batches", type=int, default=None,
                   help="limit val images (quick check; batch_size is 1)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    # Default the weights path to the folder matching the neck choice:
    # outputs/ for v1 (largefov), outputsv2/ for v2 (aspp).
    if args.weights is None:
        out_dir = config.OUTPUT_DIR_V2 if args.neck == "aspp" else config.OUTPUT_DIR
        args.weights = os.path.join(out_dir, "best.pt")

    # Val set = VOC2012 seg val at original resolution (padded to /8),
    # batch_size=1 because every image keeps its own size.
    val_set = VOCSegDataset(image_set="val", train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Val images: {len(val_set)}")

    # Build model and load weights (pretrained=False: the checkpoint already
    # contains trained weights, no need to fetch ImageNet ones first).
    # Pass both v1 and v2 neck params; DeepLab uses only the ones that match
    # neck_type (largefov reads hidden_channels/atrous_rate, aspp reads rates).
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=False, backbone=config.BACKBONE,
                    neck_type=args.neck,
                    neck_hidden_channels=config.NECK_HIDDEN_CHANNELS,
                    neck_out_channels=config.NECK_OUT_CHANNELS,
                    atrous_rate=config.ATROUS_RATE,
                    aspp_rates=config.ASPP_RATES,
                    neck_dropout=config.NECK_DROPOUT).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights: {args.weights}")

    compute_miou(model, val_loader, device, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
