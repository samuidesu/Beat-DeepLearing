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
    p.add_argument("--neck", choices=["largefov", "aspp", "aspp_v3"],
                   default="largefov",
                   help="neck the checkpoint was trained with: largefov (v1, "
                        "train.py), aspp (v2, trainv2.py) or aspp_v3 (v3, "
                        "trainV3.py). MUST match the checkpoint.")
    p.add_argument("--multi-grid", action="store_true",
                   help="the checkpoint used the Multi-Grid backbone (trainV3 "
                        "--multi-grid). MUST match: Multi-Grid changes only "
                        "dilation attributes, not weight shapes, so a mismatch "
                        "loads silently but evaluates with the wrong dilations.")
    p.add_argument("--output-stride", type=int, default=config.OUTPUT_STRIDE,
                   choices=[8, 16],
                   help="backbone output stride the checkpoint used "
                        "(only with --multi-grid)")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in the folder "
                        "matching --neck / --multi-grid)")
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
    # outputs/ (v1), outputsv2/ (v2), outputsv3/ (v3). A Multi-Grid v3 run
    # lives in outputsv3_mg/ (the "_mg" suffix trainV3 adds).
    if args.weights is None:
        out_dir = {
            "largefov": config.OUTPUT_DIR,
            "aspp": config.OUTPUT_DIR_V2,
            "aspp_v3": config.OUTPUT_DIR_V3,
        }[args.neck]
        if args.neck == "aspp_v3" and args.multi_grid:
            out_dir += "_mg"
        args.weights = os.path.join(out_dir, "best.pt")

    # Val set = VOC2012 seg val at original resolution (padded to /8),
    # batch_size=1 because every image keeps its own size.
    val_set = VOCSegDataset(image_set="val", train=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=args.num_workers)
    print(f"Val images: {len(val_set)}")

    # Build model and load weights (pretrained=False: the checkpoint already
    # contains trained weights, no need to fetch ImageNet ones first).
    # Pass every neck's params; DeepLab uses only the ones matching neck_type
    # (largefov -> hidden_channels/atrous_rate, aspp -> aspp_rates,
    # aspp_v3 -> aspp_v3_rates/aspp_v3_hidden).
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=False, backbone=config.BACKBONE,
                    neck_type=args.neck,
                    neck_hidden_channels=config.NECK_HIDDEN_CHANNELS,
                    neck_out_channels=config.NECK_OUT_CHANNELS,
                    atrous_rate=config.ATROUS_RATE,
                    aspp_rates=config.ASPP_RATES,
                    aspp_v3_rates=config.ASPP_V3_RATES,
                    aspp_v3_hidden=config.ASPP_V3_HIDDEN,
                    neck_dropout=config.NECK_DROPOUT,
                    multi_grid=args.multi_grid,
                    output_stride=args.output_stride,
                    block4_multi_grid=config.BLOCK4_MULTI_GRID).to(device)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded weights: {args.weights}")

    compute_miou(model, val_loader, device, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
