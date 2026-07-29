"""Training entry point for DeepLab-v3 (ASPP + image-level pooling) on VOC 2012.

Third in the DeepLab progression, changing only the neck each time:
    v1 (train.py)   -- one LargeFOV atrous conv (a single field of view).
    v2 (trainv2.py) -- ASPP: parallel atrous convs emitting SCORES, summed.
    v3 (this file)  -- ASPP whose branches emit FEATURES, PLUS an image-level
                       global-pooling branch; all concatenated and fused by a
                       head. Adds whole-image context on top of v2's multi-
                       scale views, and lets the head learn each branch's
                       weight instead of a fixed sum.

Everything else (dilated ResNet backbone, per-pixel cross-entropy loss, the
two-stage layered-LR finetune, the eval protocol) is IDENTICAL to v1/v2, so
the three logs are directly comparable -- a clean neck-only ablation.

Like trainv2.py, this file IMPORTS every reusable piece from train.py and
overrides only:
    * the model         -> DeepLab(neck_type="aspp_v3", ...)
    * the output folder  -> config.OUTPUT_DIR_V3 (outputsv3/, its own folder)

Optional Multi-Grid backbone (DeepLab-v3's layer4 block-level atrous rates):
    --multi-grid          swaps in model/backbone_v3_multigrid.py. OFF by
                          default, so a plain `python trainV3.py` reproduces the
                          exact same experiment as before. Multi-Grid runs write
                          to outputsv3_mg/ so they never clobber the plain-v3
                          results.
    --output-stride 8|16  backbone output stride (only meaningful with
                          --multi-grid; the plain backbone is always 8).

Stage 2 is skippable: --epochs-stage2 0 trains stage 1 only (the stage blocks
are guarded by `epochs > 0`).

Run it:
    python trainV3.py                            # plain v3, config defaults
    python trainV3.py --epochs-stage1 32 --epochs-stage2 30
    python trainV3.py --multi-grid               # v3 + Multi-Grid layer4
    python trainV3.py --multi-grid --epochs-stage2 0   # MG, stage 1 only
    python trainV3.py --download --device cpu

Compare afterwards: outputs/ (v1) vs. outputsv2/ (v2) vs. outputsv3/ (v3)
vs. outputsv3_mg/ (v3 + Multi-Grid).
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
# v1/v2/v3 differ only in the two lines noted in the docstring. (run_stage
# already logs every param group's LR, so the v3 log gets that for free.)
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
        description="Train DeepLab-v3 (ASPP + global pooling) on PASCAL VOC 2012")
    p.add_argument("--download", action="store_true", help="download VOC before training")
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--epochs-stage1", type=int, default=config.STAGE1_EPOCHS)
    p.add_argument("--epochs-stage2", type=int, default=config.STAGE2_EPOCHS,
                   help="stage-2 epochs; 0 skips stage 2 (train stage 1 only)")
    p.add_argument("--multi-grid", action="store_true",
                   help="use the Multi-Grid layer4 backbone "
                        "(backbone_v3_multigrid.py); OFF = plain dilated backbone")
    p.add_argument("--output-stride", type=int, default=config.OUTPUT_STRIDE,
                   choices=[8, 16],
                   help="backbone output stride (only used with --multi-grid)")
    p.add_argument("--output-dir", default=None,
                   help="output folder name/path. Default: outputsv3/ (or "
                        "outputsv3_mg/ with --multi-grid). A bare name is placed "
                        "under the project root; an absolute path is used as-is.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)

    # Output folder. --output-dir overrides everything (a bare name lands under
    # the project root, next to outputs*/; an absolute path is used verbatim).
    # Otherwise: v3 writes to its OWN folder so it never clobbers v1/v2, and
    # Multi-Grid runs get the "_mg" suffix so plain-v3 and MG sit side by side.
    if args.output_dir:
        output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                      else os.path.join(config.PROJECT_ROOT, args.output_dir))
    else:
        output_dir = config.OUTPUT_DIR_V3 + ("_mg" if args.multi_grid else "")
    os.makedirs(output_dir, exist_ok=True)
    variant = "Multi-Grid backbone" if args.multi_grid else "plain dilated backbone"
    print(f"Device: {device}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Output dir: {output_dir}  (DeepLab-v3, ASPP + global pooling, {variant})")

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device)
    print(f"Train batches: {len(train_loader)}  Val images: {len(val_loader)}")

    # ---- Model + loss ----
    # Changes vs. train.py: neck_type="aspp_v3" and the optional Multi-Grid
    # backbone. When --multi-grid is off, multi_grid=False makes DeepLab build
    # the plain backbone -> identical to the earlier v3 experiment.
    model = DeepLab(num_classes=config.NUM_CLASSES,
                    pretrained=True, backbone=config.BACKBONE,
                    neck_type="aspp_v3",
                    aspp_v3_rates=config.ASPP_V3_RATES,
                    aspp_v3_hidden=config.ASPP_V3_HIDDEN,
                    neck_dropout=config.NECK_DROPOUT,
                    multi_grid=args.multi_grid,
                    output_stride=args.output_stride,
                    block4_multi_grid=config.BLOCK4_MULTI_GRID).to(device)
    criterion = DeepLabLoss(ignore_index=config.IGNORE_INDEX).to(device)
    print(f"ASPP-v3 rates: {config.ASPP_V3_RATES} (+ 1x1 + global pooling)  "
          f"hidden: {config.ASPP_V3_HIDDEN}")
    if args.multi_grid:
        print(f"Multi-Grid: layer4 units {config.BLOCK4_MULTI_GRID}  "
              f"output_stride {args.output_stride}  "
              f"-> layer4 dilations {model.backbone.block4_actual_dilations}")

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
    neck_meta = {"neck_type": "aspp_v3",
                 "aspp_v3_rates": list(config.ASPP_V3_RATES),
                 "aspp_v3_hidden": config.ASPP_V3_HIDDEN,
                 "multi_grid": args.multi_grid}
    if args.multi_grid:
        # Record the ACTUAL layer4 dilations so the log is self-explaining.
        neck_meta["output_stride"] = args.output_stride
        neck_meta["block4_multi_grid"] = list(config.BLOCK4_MULTI_GRID)
        neck_meta["block4_actual_dilations"] = list(model.backbone.block4_actual_dilations)
    meta = collect_run_meta(
        args, device, train_loader, val_loader,
        model_name="DeepLab-v3 (ASPP + global pooling)"
                   + (" + Multi-Grid" if args.multi_grid else ""),
        neck=neck_meta)
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
