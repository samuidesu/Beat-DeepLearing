"""Training entry point for DeepLab-v2 (ASPP neck) on PASCAL VOC 2012.

This is the v1 train.py with ONE thing changed: the neck. v1 used a single
LargeFOV atrous conv (one field of view); v2 uses ASPP -- several parallel
atrous convs at different rates, summed -- so the network sees each object at
multiple scales at once. Everything else (dilated ResNet backbone, 1x1 head,
per-pixel cross-entropy loss, two-stage layered-LR finetune, eval protocol)
is IDENTICAL, which is the whole point: a clean single-variable comparison of
LargeFOV vs. ASPP.

To keep that comparison honest AND avoid copy-pasting a whole training loop,
this file IMPORTS every reusable piece from train.py and overrides only:
    * the model         -> DeepLab(neck_type="aspp", aspp_rates=config.ASPP_RATES)
    * the output folder  -> config.OUTPUT_DIR_V2 (so v1's outputs/ is untouched)

Run it exactly like train.py:
    python trainv2.py                       # config.py defaults
    python trainv2.py --epochs-stage1 32 --epochs-stage2 30
    python trainv2.py --download --device cpu

Compare afterwards: outputs/ (v1) vs. outputsv2/ (v2) -- best.pt, the
training_log.json, and the loss/mIoU curves sit side by side.
"""

import os
import argparse

import torch
import torch.optim as optim

import config
from model.deeplab import DeepLab
from losses.deeplab_loss import DeepLabLoss
from utils.metrics import compute_miou

# Reuse the ENTIRE training pipeline from train.py -- unchanged machinery, so
# v1 and v2 differ only in the two lines noted in the docstring. (run_stage
# already logs every param group's LR, so the v2 log gets that for free.)
from train import (
    set_seed,
    get_device,
    build_dataloaders,
    run_stage,
    save_log,
    collect_run_meta,
    summarize_result,
    plot_curves,
    build_layered_optimizer,
    count_trainable,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train DeepLab-v2 (ASPP) on PASCAL VOC 2012 segmentation")
    p.add_argument("--download", action="store_true", help="download VOC before training")
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--epochs-stage1", type=int, default=config.STAGE1_EPOCHS)
    p.add_argument("--epochs-stage2", type=int, default=config.STAGE2_EPOCHS)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)

    # v2 writes to its OWN folder so it never clobbers v1's best.pt / log / curves.
    output_dir = config.OUTPUT_DIR_V2
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Output dir: {output_dir}  (DeepLab-v2, ASPP neck)")

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device)
    print(f"Train batches: {len(train_loader)}  Val images: {len(val_loader)}")

    # ---- Model + loss ----
    # The ONLY substantive change vs. train.py: neck_type="aspp" (+ the rates).
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=True, backbone=config.BACKBONE,
                    neck_type="aspp",
                    neck_out_channels=config.NECK_OUT_CHANNELS,
                    aspp_rates=config.ASPP_RATES,
                    neck_dropout=config.NECK_DROPOUT).to(device)
    criterion = DeepLabLoss(ignore_index=config.IGNORE_INDEX).to(device)
    print(f"ASPP rates: {config.ASPP_RATES}")

    history = []
    best = {"miou": -1.0, "epoch": -1}

    # ---- Stage 1: freeze LOW backbone, train dilated high stages + neck/head ----
    if args.epochs_stage1 > 0:
        print("\n=== Stage 1: freeze stem/layer1/layer2, "
              "train layer3/layer4 + neck + head ===")
        model.freeze_backbone_low()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        # Two LR tiers: from-scratch neck/head fast, pretrained-but-modified
        # layer3/4 slower. backbone_low is frozen -> its tier is skipped.
        optimizer = build_layered_optimizer(
            model,
            lr_head=config.STAGE1_LR_HEAD,
            lr_backbone_high=config.STAGE1_LR_BACKBONE_HIGH,
            lr_backbone_low=None,
            weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage1)
        best = run_stage(1, model, train_loader, val_loader, criterion, optimizer,
                         scheduler, args.epochs_stage1, device, history, best, output_dir)

    # ---- Stage 2: unfreeze the whole backbone, layered-LR finetune ----
    if args.epochs_stage2 > 0:
        print("\n=== Stage 2: unfreeze ALL, layered-LR finetune ===")
        model.unfreeze_backbone_all()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        # Three LR tiers: the deeper into pristine pretrained territory,
        # the gentler the updates.
        optimizer = build_layered_optimizer(
            model,
            lr_head=config.STAGE2_LR_HEAD,
            lr_backbone_high=config.STAGE2_LR_BACKBONE_HIGH,
            lr_backbone_low=config.STAGE2_LR_BACKBONE_LOW,
            weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)
        best = run_stage(2, model, train_loader, val_loader, criterion, optimizer,
                         scheduler, args.epochs_stage2, device, history, best, output_dir)

    # ---- Save logs + curves ----
    # Snapshot the config + training params alongside the per-epoch history.
    # Only difference vs. train.py: the model name + the neck-specific fields.
    meta = collect_run_meta(
        args, device, train_loader, val_loader,
        model_name="DeepLab-v2 (ASPP)",
        neck={"neck_type": "aspp", "aspp_rates": list(config.ASPP_RATES)})
    meta["best_proxy"] = {"miou": round(best["miou"], 4), "epoch": best["epoch"]}
    save_log(history, output_dir, meta)
    plot_curves(history, output_dir)
    print(f"\nDone. Best mIoU={best['miou']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {output_dir}")

    # ---- Final FULL mIoU on the best checkpoint (all 1449 val images) ----
    best_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("\nComputing full mIoU on VOC2012 val (best checkpoint)...")
        result = compute_miou(model, val_loader, device)  # verbose: per-class table
        # Record the full-set result in the log too, then re-save.
        meta["final_full_val"] = summarize_result(result)
        save_log(history, output_dir, meta)


if __name__ == "__main__":
    main()
