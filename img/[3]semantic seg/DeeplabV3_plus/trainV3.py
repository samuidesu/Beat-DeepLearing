"""Training entry point for DeepLab-v3+ (ASPP encoder + decoder) on VOC 2012.

DeepLab-v3+ = v3's ASPP encoder + a small DECODER. v3 upsampled the stride-8
logits straight to full resolution (an 8x jump), blurring boundaries; v3+ fuses
the encoder output with a crisp stride-4 low-level backbone feature (c2) in a
decoder, then upsamples only 4x -- sharper edges for thin structures. See
model/neckV3.py (encoder), model/headV3.py (decoder), model/deeplab.py (wiring).

This file IMPORTS the shared training machinery from train.py (data, the
two-stage layered-LR loop, logging, plotting) and only builds the v3+ model and
writes to its own output folder, so the pipeline stays identical to the earlier
DeepLab experiments -- a clean model-only comparison.

Stage 2 is skippable: --epochs-stage2 0 trains stage 1 only (the stage blocks
are guarded by `epochs > 0`).

Run it:
    python trainV3.py                            # config.py defaults
    python trainV3.py --epochs-stage1 32 --epochs-stage2 30
    python trainV3.py --epochs-stage2 0          # stage 1 only
    python trainV3.py --output-dir my_run        # custom output folder
    python trainV3.py --download --device cpu
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
# only the model differs from the other DeepLab experiments. (run_stage already
# logs every param group's LR, so the v3+ log gets that for free.)
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
        description="Train DeepLab-v3+ (ASPP encoder + decoder) on PASCAL VOC 2012")
    p.add_argument("--download", action="store_true", help="download VOC before training")
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--epochs-stage1", type=int, default=config.STAGE1_EPOCHS)
    p.add_argument("--epochs-stage2", type=int, default=config.STAGE2_EPOCHS,
                   help="stage-2 epochs; 0 skips stage 2 (train stage 1 only)")
    p.add_argument("--output-dir", default=None,
                   help="output folder name/path. Default: outputsv3plus/. A bare "
                        "name is placed under the project root; an absolute path "
                        "is used as-is.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)

    # Output folder. --output-dir overrides the default (a bare name lands under
    # the project root, next to outputs*/; an absolute path is used verbatim).
    if args.output_dir:
        output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                      else os.path.join(config.PROJECT_ROOT, args.output_dir))
    else:
        output_dir = config.OUTPUT_DIR_V3PLUS
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Output dir: {output_dir}  (DeepLab-v3+, ASPP encoder + decoder)")

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device)
    print(f"Train batches: {len(train_loader)}  Val images: {len(val_loader)}")

    # ---- Model + loss ----
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=True, backbone=config.BACKBONE,
                    aspp_rates=config.ASPP_V3_RATES,
                    aspp_hidden=config.ASPP_V3_HIDDEN,
                    low_level_proj=config.LOW_LEVEL_PROJ,
                    dropout=config.NECK_DROPOUT).to(device)
    criterion = DeepLabLoss(ignore_index=config.IGNORE_INDEX).to(device)
    print(f"ASPP rates: {config.ASPP_V3_RATES} (+ 1x1 + global pooling)  "
          f"hidden: {config.ASPP_V3_HIDDEN}  low-level proj: {config.LOW_LEVEL_PROJ}")

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
    # Skipped entirely when --epochs-stage2 0: the best checkpoint is then
    # whatever stage 1 produced (frozen low backbone).
    if args.epochs_stage2 <= 0:
        print("\n=== Stage 2 skipped (--epochs-stage2 0) ===")
    else:
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
    meta = collect_run_meta(
        args, device, train_loader, val_loader,
        model_name="DeepLab-v3+ (ASPP encoder + decoder)",
        neck={"model": "deeplabv3plus",
              "aspp_rates": list(config.ASPP_V3_RATES),
              "aspp_hidden": config.ASPP_V3_HIDDEN,
              "low_level_proj": config.LOW_LEVEL_PROJ})
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
