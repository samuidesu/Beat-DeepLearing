"""Training entry point for RNN text classification on SST-2 (GLUE).

The recurrent CELL is selectable and each cell owns its output folder:
    --cell rnn   -> vanilla Elman RNN -> outputs_rnn/
    --cell lstm  -> LSTM              -> outputs_lstm/
    --cell gru   -> GRU               -> outputs_gru/
so rnn-vs-lstm-vs-gru is a clean single-variable comparison sharing everything
else (vocabulary, GloVe init, pooling, loss, schedule, eval protocol).

Two-stage finetuning (same shape as the segmentation experiments, with the
GloVe embedding table playing the pretrained-backbone role):
    Stage 1 - FREEZE the word vectors; the from-scratch encoder + head learn
              to read fixed GloVe features. Both trainable parts are random,
              so they share one LR tier.
    Stage 2 - unfreeze the embedding and finetune everything with LAYERED
              learning rates: embedding slowest (it already knows English),
              encoder middle, head fastest.

NOTE on pretrained vectors: with --no-glove (or config.USE_GLOVE = False) the
word vectors start random -- then skip stage 1 (--epochs-stage1 0), because
freezing RANDOM embeddings means training an encoder to read noise.

This file wires up the full pipeline: data -> model -> loss -> optimize, plus
evaluation (dev loss + accuracy + macro-F1), checkpointing, logging
(training_log.json) and curve plotting -- deliberately parallel to the HRNet
and DeepLab train.py files.

Usage:
    python train.py                        # LSTM, config.py defaults
    python train.py --cell rnn             # vanilla RNN -> outputs_rnn/
    python train.py --download             # fetch SST-2 + GloVe first
    python train.py --no-glove --epochs-stage1 0   # from-scratch embeddings
    python train.py --pooling max --batch-size 32 --device cpu
"""

import os
import json
import time
import random
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import config
from model.rnn_classifier import RNNClassifier
from losses.cls_loss import ClassificationLoss
from dataset.sst2 import (
    SST2Dataset,
    build_vocab_from_train,
    collate_batch,
    download_sst2,
    sst2_present,
)
from dataset.glove import build_embedding_matrix, download_glove, glove_present
from utils.metrics import ConfusionMatrix, compute_accuracy
from utils.viz import plot_confusion_matrix

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to a no-op wrapper.

    def tqdm(iterable, **kwargs):
        return iterable


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int = 42):
    """Seed python / numpy / torch RNGs for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN's RNN kernels are the nondeterministic ones here (autotuned
    # algorithms, atomics in the backward pass). Forcing the deterministic
    # path costs some speed and buys reproducible curves.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Seed each DataLoader worker (only relevant when num_workers > 0)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(pref: str = "auto") -> torch.device:
    """Pick a device: explicit `pref`, else cuda > mps > cpu."""
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataloaders(batch_size, num_workers, download, device, use_glove):
    """Build the train / dev dataloaders and the vocabulary.

    Train = SST-2 train.tsv (67,349 phrases), shuffled.
    Dev   = SST-2 dev.tsv (872 sentences), in order, larger batches (no
            backward pass to fit in memory).
    Both use collate_batch, which pads each batch to ITS OWN longest sentence.

    Output:
        (train_loader, dev_loader, vocab).
    """
    # Fetch data when asked (--download) OR when anything is missing, so a
    # fresh machine needs no separate step. Both fetches are idempotent.
    if download or not sst2_present():
        download_sst2()
    if use_glove and (download or not glove_present()):
        download_glove()

    vocab = build_vocab_from_train()
    train_set = SST2Dataset("train", vocab)
    dev_set = SST2Dataset("dev", vocab)

    pin = device.type == "cuda"  # pinned memory only helps CUDA copies
    g = torch.Generator()
    g.manual_seed(config.SEED)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin,
    )
    return train_loader, dev_loader, vocab


# -----------------------------------------------------------------------------
# Train / evaluate one epoch
# -----------------------------------------------------------------------------
def _accumulate(running: dict, items: dict, n: int):
    """Add this batch's loss components (weighted by batch size) into `running`."""
    for k, v in items.items():
        running[k] = running.get(k, 0.0) + v * n


def _average(running: dict, total: int) -> dict:
    """Turn summed loss components into per-sample averages."""
    return {k: (v / max(total, 1)) for k, v in running.items()}


def train_one_epoch(model, loader, criterion, optimizer, device, epoch_desc="", grad_clip=None):
    """Run one training epoch.

    Input:
        model, loader, criterion, optimizer, device as usual.
        epoch_desc: string shown on the progress bar.
        grad_clip: max global gradient norm (config.GRAD_CLIP), or None.

    Output:
        dict of average loss components over the epoch ({"loss"} here --
        cross-entropy is a single term).
    """
    model.train()
    running, seen = {}, 0
    for ids, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        ids = ids.to(device, non_blocking=True)  # [B, L]
        lengths = lengths.to(device, non_blocking=True)  # [B]
        labels = labels.to(device, non_blocking=True)  # [B]
        bs = ids.size(0)

        logits = model(ids, lengths)  # [B, 2]
        loss, items = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # RNN-specific and NOT optional: backprop through time can produce a
        # single enormous gradient that undoes an epoch of progress. Clipping
        # rescales the whole gradient (preserving its direction) whenever its
        # norm exceeds the threshold.
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        _accumulate(running, items, bs)
        seen += bs

    return _average(running, seen)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch_desc="eval"):
    """One pass over the dev set -> average loss AND accuracy / macro-F1.

    The whole dev split is 872 sentences, so -- unlike the segmentation
    projects, which evaluated a capped proxy every epoch -- this is the FULL
    metric every epoch, and the best-checkpoint choice is made on the real
    number.

    Output:
        dict {"loss": dev_loss, "accuracy": ..., "macro_f1": ...}.
    """
    model.eval()
    cm = ConfusionMatrix(config.NUM_CLASSES)
    running, seen = {}, 0
    for ids, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        ids = ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bs = ids.size(0)

        logits = model(ids, lengths)
        _, items = criterion(logits, labels)
        cm.update(logits.argmax(dim=1), labels)

        _accumulate(running, items, bs)
        seen += bs

    out = _average(running, seen)
    res = cm.compute()
    out["accuracy"] = res["accuracy"]
    out["macro_f1"] = res["macro_f1"]
    return out


# -----------------------------------------------------------------------------
# Stage runner
# -----------------------------------------------------------------------------
def run_stage(
    stage_id,
    model,
    train_loader,
    dev_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    device,
    history,
    best,
    ckpt_dir,
):
    """Train for `epochs` epochs, logging + checkpointing each one.

    Input:
        stage_id: 1 or 2 (recorded in the history for plotting).
        history: list of per-epoch dict records, appended in place.
        best: dict {"accuracy": float, "epoch": int} tracking the best model.
        ckpt_dir: directory to save best.pt / last.pt.

    Output:
        the (possibly updated) `best` dict.
    """
    for e in range(1, epochs + 1):
        # Global epoch number = epochs already recorded + 1.
        global_epoch = len(history) + 1
        t0 = time.time()

        desc = f"[stage {stage_id}] epoch {e}/{epochs}"
        # Read the LRs BEFORE training/stepping so they reflect the LRs
        # actually used this epoch. With layered LRs there are 2-3 param
        # groups, each on its own cosine schedule; log EVERY group.
        group_lrs = {
            g.get("name", f"group{i}"): g["lr"] for i, g in enumerate(optimizer.param_groups)
        }
        lr = optimizer.param_groups[0]["lr"]  # head tier; kept for back-compat
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, desc, grad_clip=config.GRAD_CLIP
        )
        dev_metrics = evaluate(model, dev_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": global_epoch,
            "stage": stage_id,
            "lr": lr,
            # One field per param group: lr_head / lr_encoder / lr_embedding.
            **{f"lr_{name}": v for name, v in group_lrs.items()},
            "time_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%m-%d %H:%M:%S"),  # wall-clock at epoch end
            **{f"train_{k}": v for k, v in train_metrics.items()},
            "val_loss": dev_metrics.get("loss", 0.0),
            "accuracy": dev_metrics["accuracy"],
            "macro_f1": dev_metrics["macro_f1"],
        }
        history.append(record)

        lr_str = " ".join(f"{name}={v:.2e}" for name, v in group_lrs.items())
        print(
            f"[{record['timestamp']}] {desc}  lr[{lr_str}]  "
            f"train_loss={train_metrics.get('loss', 0):.4f}  "
            f"val_loss={dev_metrics.get('loss', 0):.4f}  "
            f"acc={dev_metrics['accuracy']:.4f}  "
            f"macroF1={dev_metrics['macro_f1']:.4f}  "
            f"({record['time_sec']}s)"
        )

        # Checkpoint: always save 'last', save 'best' on ACCURACY improvement
        # (SST-2's official metric). Selecting on dev loss would be a
        # different -- and with label smoothing, a slightly misaligned --
        # criterion; the two disagree late in training once the model starts
        # overfitting confidence rather than correctness.
        # torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pt"))
        cur = dev_metrics["accuracy"]
        if cur > best["accuracy"]:
            best["accuracy"] = cur
            best["epoch"] = global_epoch
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

    return best


# -----------------------------------------------------------------------------
# Logging / plotting
# -----------------------------------------------------------------------------
def save_log(history, output_dir, meta=None):
    """Write the run to outputs_*/training_log.json.

    With `meta` (the config + training-param snapshot from collect_run_meta),
    the file is {"meta": {...}, "history": [...per-epoch...]} so a run is
    fully reproducible from its log alone.
    """
    payload = {"meta": meta, "history": history} if meta is not None else history
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(payload, f, indent=2)


def collect_run_meta(args, device, train_loader, dev_loader, vocab, *, model_name, extra=None):
    """Snapshot the run's config + training params for training_log.json.

    Records enough to reproduce the run from the log alone: corpus/vocab
    stats, model hyperparams, and the two-stage layered-LR schedule.
    """
    meta = {
        "model": model_name,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "seed": config.SEED,
        "data": {
            "data_root": config.DATA_ROOT,
            "train_sentences": len(train_loader.dataset),
            "train_batches": len(train_loader),
            "dev_sentences": len(dev_loader.dataset),
            "batch_size": args.batch_size,
            "eval_batch_size": config.EVAL_BATCH_SIZE,
            "num_workers": args.num_workers,
            "vocab_size": len(vocab),
            "min_freq": config.MIN_FREQ,
            "max_len": config.MAX_LEN,
            "label_counts": train_loader.dataset.label_counts(),
            "dev_unk_rate": round(dev_loader.dataset.unk_rate(), 4),
        },
        "model_cfg": {
            "cell": args.cell,
            "embed_dim": config.EMBED_DIM,
            "hidden_size": config.HIDDEN_SIZE,
            "num_layers": config.NUM_LAYERS,
            "bidirectional": config.BIDIRECTIONAL,
            "pooling": args.pooling,
            "dropout": config.DROPOUT,
            "num_classes": config.NUM_CLASSES,
        },
        "optim": {
            "optimizer": "Adam",
            "weight_decay": config.WEIGHT_DECAY,
            "grad_clip": config.GRAD_CLIP,
            "label_smoothing": config.LABEL_SMOOTHING,
            "epochs_stage1": args.epochs_stage1,
            "epochs_stage2": args.epochs_stage2,
            "stage1_lr_head": config.STAGE1_LR_HEAD,
            "stage1_lr_encoder": config.STAGE1_LR_ENCODER,
            "stage2_lr_head": config.STAGE2_LR_HEAD,
            "stage2_lr_encoder": config.STAGE2_LR_ENCODER,
            "stage2_lr_embedding": config.STAGE2_LR_EMBEDDING,
        },
    }
    if extra:
        meta["model_cfg"].update(extra)
    return meta


def summarize_result(result):
    """Compact a compute_accuracy() dict for the log."""
    return {
        "accuracy": round(result["accuracy"], 4),
        "macro_f1": round(result["macro_f1"], 4),
        "per_class": {
            name: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
            for name, row in zip(config.CLASS_NAMES, result["per_class"])
        },
        "matrix": result["matrix"],
    }


def plot_curves(history, output_dir, title_prefix="RNN"):
    """Plot training curves from the history and save PNGs to output_dir.

    Produces:
        loss_curve.png - train vs. dev cross-entropy loss per epoch.
        acc_curve.png  - dev accuracy and macro-F1 per epoch.
    A dashed vertical line marks the stage-1 -> stage-2 boundary.
    """
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    stage2_start = next((r["epoch"] for r in history if r["stage"] == 2), None)

    # ---- Figure 1: cross-entropy loss, train vs dev ----
    plt.figure(figsize=(8, 5))
    # Fall back to the former *_total names so existing logs can still be plotted.
    train_losses = [r.get("train_loss", r.get("train_total", 0)) for r in history]
    val_losses = [r.get("val_loss", r.get("val_total", 0)) for r in history]
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="dev")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("cross-entropy loss")
    plt.title(f"{title_prefix} loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # ---- Figure 2: dev accuracy + macro-F1 ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r.get("accuracy", 0) for r in history], label="accuracy")
    plt.plot(epochs, [r.get("macro_f1", 0) for r in history], label="macro F1")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    # Zoomed y-range: a binary classifier starts at ~0.5, so plotting from 0
    # would squash every difference that matters into a thin band.
    plt.ylim(0.5, 1.0)
    plt.title(f"{title_prefix} dev accuracy / macro F1")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "acc_curve.png"), dpi=150)
    plt.close()


# -----------------------------------------------------------------------------
# Optimizer builder (layered learning rates)
# -----------------------------------------------------------------------------
def build_layered_optimizer(model, lr_head, lr_encoder, lr_embedding, weight_decay):
    """Build an Adam optimizer with up to three layered param groups.

    The split comes from model.parameter_groups() (frozen params excluded):

        head      -> lr_head      (from-scratch classifier, fastest)
        encoder   -> lr_encoder   (from-scratch RNN, same tier in stage 1)
        embedding -> lr_embedding (pretrained GloVe, slowest)

    Input:
        lr_*: learning rate per tier. Pass None to skip a tier (stage 1 passes
            lr_embedding=None -- those params are frozen anyway, and an empty
            param group would confuse the scheduler printout).
        weight_decay: shared across all groups.

    Output:
        torch.optim.Adam with one param_group per ACTIVE tier. Group order is
        head first, so optimizer.param_groups[0]["lr"] (what run_stage logs as
        the scalar "lr") is always the head LR.
    """
    groups = model.parameter_groups()
    param_groups = []
    for name, lr in (
        ("head", lr_head),
        ("encoder", lr_encoder),
        ("embedding", lr_embedding),
    ):
        params = groups[name]
        if lr is None or not params:
            continue
        param_groups.append({"params": params, "lr": lr, "name": name})
    return optim.Adam(param_groups, weight_decay=weight_decay)


def count_trainable(model):
    """Return the number of trainable parameters (in millions)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train an RNN sentence classifier on SST-2")
    p.add_argument(
        "--download", action="store_true", help="download SST-2 (and GloVe) before training"
    )
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument(
        "--epochs-stage1",
        type=int,
        default=config.STAGE1_EPOCHS,
        help="stage-1 epochs (frozen embeddings); 0 skips stage 1 "
        "(use 0 together with --no-glove)",
    )
    p.add_argument(
        "--epochs-stage2",
        type=int,
        default=config.STAGE2_EPOCHS,
        help="stage-2 epochs (embeddings unfrozen); 0 skips stage 2",
    )
    p.add_argument(
        "--cell",
        choices=["rnn", "lstm", "gru"],
        default=config.CELL,
        help="recurrent cell: rnn (vanilla) / lstm / gru. Each "
        "writes to its own outputs_<cell>/ folder",
    )
    p.add_argument(
        "--pooling",
        choices=["last", "max", "mean"],
        default=config.POOLING,
        help="how token features collapse into a sentence vector",
    )
    p.add_argument(
        "--no-glove", action="store_true", help="train word vectors from scratch (no GloVe init)"
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="output folder name/path. Default: outputs_<cell>/. A "
        "bare name is placed under the project root; an "
        "absolute path is used as-is.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(config.SEED)
    device = get_device(args.device)
    use_glove = config.USE_GLOVE and not args.no_glove

    # Output folder: one per cell so the three experiments never overwrite
    # each other; --output-dir overrides (used for the --no-glove ablation).
    if args.output_dir:
        output_dir = (
            args.output_dir
            if os.path.isabs(args.output_dir)
            else os.path.join(config.PROJECT_ROOT, args.output_dir)
        )
    else:
        output_dir = config.output_dir_for_cell(args.cell)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}")
    print(f"Data root: {config.DATA_ROOT}")
    print(
        f"Output dir: {output_dir}  ({args.cell} cell, {args.pooling} pooling, "
        f"{'GloVe' if use_glove else 'scratch'} embeddings)"
    )

    # ---- Data ----
    train_loader, dev_loader, vocab = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device, use_glove
    )
    print(
        f"Train batches: {len(train_loader)}  Dev sentences: {len(dev_loader.dataset)}  "
        f"Vocab: {len(vocab)}"
    )
    # Save the vocabulary NEXT TO THE CHECKPOINT. The build is deterministic,
    # so eval.py could rebuild it -- but a checkpoint whose id mapping lives
    # only in a rebuild step is a checkpoint waiting to be silently misread.
    vocab.save(os.path.join(output_dir, "vocab.json"))

    # ---- Model + loss ----
    vectors = None
    if use_glove:
        vectors, n_found = build_embedding_matrix(vocab)
    model = RNNClassifier(
        vocab_size=len(vocab),
        num_classes=config.NUM_CLASSES,
        embed_dim=config.EMBED_DIM,
        hidden_size=config.HIDDEN_SIZE,
        cell=args.cell,
        num_layers=config.NUM_LAYERS,
        bidirectional=config.BIDIRECTIONAL,
        pooling=args.pooling,
        dropout=config.DROPOUT,
        pad_idx=config.PAD_IDX,
        pretrained_vectors=vectors,
    ).to(device)
    criterion = ClassificationLoss(label_smoothing=config.LABEL_SMOOTHING).to(device)
    print(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"(embedding {len(vocab) * config.EMBED_DIM / 1e6:.2f}M)"
    )

    # From-scratch guard: stage 1 freezes the embedding, which only makes
    # sense when it is pretrained. Warn (don't override) so the run stays
    # reproducible from the command line alone.
    if not model.embedding.pretrained_loaded and args.epochs_stage1 > 0:
        print(
            "[SST-2] WARNING: no GloVe vectors were loaded but stage 1 will "
            "freeze the (random) embedding table. Consider "
            "--epochs-stage1 0 for from-scratch training."
        )

    history = []
    best = {"accuracy": -1.0, "epoch": -1}

    # ---- Stage 1: freeze the embedding, train encoder + head ----
    if args.epochs_stage1 > 0:
        print("\n=== Stage 1: freeze embeddings, train encoder + head ===")
        model.freeze_embedding()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_layered_optimizer(
            model,
            lr_head=config.STAGE1_LR_HEAD,
            lr_encoder=config.STAGE1_LR_ENCODER,
            lr_embedding=None,  # frozen -> tier skipped
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage1)
        best = run_stage(
            1,
            model,
            train_loader,
            dev_loader,
            criterion,
            optimizer,
            scheduler,
            args.epochs_stage1,
            device,
            history,
            best,
            output_dir,
        )

    # ---- Stage 2: unfreeze the embedding, layered-LR finetune ----
    if args.epochs_stage2 <= 0:
        print("\n=== Stage 2 skipped (--epochs-stage2 0) ===")
    else:
        print("\n=== Stage 2: unfreeze embeddings, layered-LR finetune ===")
        model.unfreeze_all()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_layered_optimizer(
            model,
            lr_head=config.STAGE2_LR_HEAD,
            lr_encoder=config.STAGE2_LR_ENCODER,
            lr_embedding=config.STAGE2_LR_EMBEDDING,
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)
        best = run_stage(
            2,
            model,
            train_loader,
            dev_loader,
            criterion,
            optimizer,
            scheduler,
            args.epochs_stage2,
            device,
            history,
            best,
            output_dir,
        )

    # ---- Save logs + curves ----
    extra = {"glove": use_glove, "glove_path": config.GLOVE_PATH if use_glove else None}
    if use_glove:
        extra["glove_coverage"] = round(n_found / len(vocab), 4)
    meta = collect_run_meta(
        args,
        device,
        train_loader,
        dev_loader,
        vocab,
        model_name=f"Bi{args.cell.upper()} ({args.pooling} pooling)",
        extra=extra,
    )
    meta["best"] = {"accuracy": round(best["accuracy"], 4), "epoch": best["epoch"]}
    save_log(history, output_dir, meta)
    plot_curves(history, output_dir, title_prefix=f"Bi{args.cell.upper()}")
    print(f"\nDone. Best dev accuracy={best['accuracy']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {output_dir}")

    # ---- Final report on the best checkpoint (per-class + confusion matrix) --
    # The per-epoch number was already the full dev set, so this does not
    # change the headline -- it adds the breakdown (and the PNG) for the
    # checkpoint we are actually keeping.
    best_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("\nBest checkpoint on SST-2 dev:")
        result = compute_accuracy(model, dev_loader, device)  # verbose report
        plot_confusion_matrix(
            result["matrix"],
            config.CLASS_NAMES,
            os.path.join(output_dir, "confusion_matrix.png"),
            title=f"Bi{args.cell.upper()} dev confusion matrix",
        )
        meta["final_dev"] = summarize_result(result)
        save_log(history, output_dir, meta)


if __name__ == "__main__":
    main()
