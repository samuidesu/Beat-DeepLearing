"""Training entry point for RNN topic classification on AG News.

The recurrent CELL is selectable and each cell owns its output folder:
    --cell rnn   -> vanilla Elman RNN -> outputs_rnn/
    --cell lstm  -> LSTM              -> outputs_lstm/
    --cell gru   -> GRU               -> outputs_gru/
so rnn-vs-lstm-vs-gru is a clean single-variable comparison sharing everything
else (vocabulary, GloVe init, pooling, loss, schedule, eval protocol).

Two-stage finetuning (same shape as the SST-2 and segmentation experiments,
with the GloVe embedding table playing the pretrained-backbone role):
    Stage 1 - FREEZE the word vectors; the from-scratch encoder + head learn
              to read fixed GloVe features. Both trainable parts are random,
              so they share one LR tier.
    Stage 2 - unfreeze the embedding and finetune everything with LAYERED
              learning rates: embedding slowest (it already knows English),
              encoder middle, head fastest.

WHAT IS EVALUATED: the 6,000-document validation split carved out of train,
NOT test.csv. AG News publishes its test labels, so scoring test every epoch
and then keeping the best epoch would quietly turn the "test accuracy" into a
best-of-N number. test.csv is read only by eval.py, once, at the end.

NOTE on pretrained vectors: with --no-glove (or config.USE_GLOVE = False) the
word vectors start random -- then skip stage 1 (--epochs-stage1 0), because
freezing RANDOM embeddings means training an encoder to read noise.

Usage:
    python train.py                        # LSTM, config.py defaults
    python train.py --cell rnn             # vanilla RNN -> outputs_rnn/
    python train.py --download             # fetch AG News + GloVe first
    python train.py --no-glove --epochs-stage1 0   # from-scratch embeddings
    python train.py --pooling max --batch-size 64 --device cpu
"""

import os
import sys
import json
import time
import random
import ctypes
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
from dataset.ag_news import (
    AGNewsDataset,
    ag_news_present,
    build_vocab_from_train,
    collate_batch,
    download_ag_news,
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


def get_device(pref: str = "auto") -> torch.device:
    """Pick a device: explicit `pref`, else cuda > mps > cpu."""
    if pref and pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_exit(code: int = 0):
    """End the process, working around a cuDNN teardown crash on Windows.

    THE BUG. On this machine (Windows 11, torch 2.11.0+cu128) a process that
    has used a cuDNN RNN with dropout in TRAIN mode dies during shutdown with

        STATUS_STACK_BUFFER_OVERRUN   0xC0000409   exit code -1073740791

    long after main() has returned. Isolated by bisection, the trigger is
    exactly cuDNN's dropout-state descriptor, which this build never releases
    cleanly:

        nn.LSTM(100, 256, num_layers=2, bidirectional=True, dropout=0.5).cuda()
        .train()  + one forward   -> crash at exit
        same but dropout=0.0                        -> exit 0
        same but .eval()                            -> exit 0
        same but torch.backends.cudnn.enabled=False -> exit 0
        same on CPU                                 -> exit 0

    That is why train.py crashed and eval.py did not: eval.py calls
    model.eval(), so the dropout descriptor is never created.

    It is NOT a bug in this project -- every checkpoint, log and PNG is
    already written and closed by the time it fires. But the nonzero exit code
    makes any script that chains runs (run_all.ps1) abort after the first
    model, so it has to be dealt with.

    THE FIX. os._exit() is not enough: on Windows it reaches ExitProcess,
    which still runs DLL_PROCESS_DETACH -- and that is where the crash lives
    (measured: still -1073740791). TerminateProcess skips DLL detach entirely
    and is the only thing that produced a 0. Everything is flushed first;
    files are already closed by their context managers.

    REJECTED ALTERNATIVES. Disabling cuDNN fixes it but costs 2.7x speed
    (measured on the project's own encoder shape, BiLSTM 2x256, batch 128,
    96 steps: 41.4 vs 112.2 ms/step) -- that is the whole training budget.
    Setting the RNN's dropout to 0 and applying nn.Dropout between manually
    stacked single-layer RNNs also works, but changes the model code for a
    problem that is purely cosmetic.
    """
    sys.stdout.flush()   # nothing below this point flushes anything
    sys.stderr.flush()
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p   # HANDLE, not int
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    os._exit(code)       # non-Windows, or if TerminateProcess somehow returns


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
def build_dataloaders(batch_size, num_workers, download, device, use_glove):
    """Build the train / val dataloaders and the vocabulary.

    Train = 114,000 documents (95% of train.csv), shuffled.
    Val   =   6,000 documents (the held-out 5%), in order, larger batches
            (no backward pass to fit in memory).
    Both use collate_batch, which pads each batch to ITS OWN longest document.

    Output:
        (train_loader, val_loader, vocab).
    """
    # Fetch data when asked (--download) OR when anything is missing, so a
    # fresh machine needs no separate step. Both fetches are idempotent.
    if download or not ag_news_present():
        download_ag_news()
    if use_glove and (download or not glove_present()):
        download_glove()

    vocab = build_vocab_from_train()
    train_set = AGNewsDataset("train", vocab)
    val_set = AGNewsDataset("val", vocab)

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
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin,
    )
    return train_loader, val_loader, vocab


# -----------------------------------------------------------------------------
# Train / evaluate one epoch
# -----------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, epoch_desc="",
                    grad_clip=None):
    """Run one training epoch.

    Input:
        model, loader, criterion, optimizer, device as usual.
        epoch_desc: string shown on the progress bar.
        grad_clip: max global gradient norm (config.GRAD_CLIP), or None.

    Output:
        float: the sample-weighted average training loss over the epoch.
    """
    model.train()
    total, seen = 0.0, 0
    for ids, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        ids = ids.to(device, non_blocking=True)          # [B, L]
        lengths = lengths.to(device, non_blocking=True)  # [B]
        labels = labels.to(device, non_blocking=True)    # [B]
        bs = ids.size(0)

        logits = model(ids, lengths)                     # [B, 4]
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        # RNN-specific and NOT optional: backprop through time can produce a
        # single enormous gradient that undoes an epoch of progress. Clipping
        # rescales the whole gradient (preserving its direction) whenever its
        # norm exceeds the threshold.
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Weight by batch size: the last batch of an epoch is usually smaller,
        # and a plain mean over batches would over-weight it.
        total += float(loss.detach()) * bs
        seen += bs

    return total / max(seen, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch_desc="eval"):
    """One pass over the validation split -> loss AND accuracy / macro-F1.

    The whole val split (6,000 documents) is scored every epoch, so the
    per-epoch number is the real number, not a capped proxy.

    Output:
        dict {"loss": ..., "accuracy": ..., "macro_f1": ...}.
    """
    model.eval()
    cm = ConfusionMatrix(config.NUM_CLASSES)
    total, seen = 0.0, 0
    for ids, lengths, labels in tqdm(loader, desc=epoch_desc, leave=False):
        ids = ids.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        bs = ids.size(0)

        logits = model(ids, lengths)
        total += float(criterion(logits, labels)) * bs
        seen += bs
        cm.update(logits.argmax(dim=1), labels)

    res = cm.compute()
    return {"loss": total / max(seen, 1),
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"]}


# -----------------------------------------------------------------------------
# Stage runner
# -----------------------------------------------------------------------------
def run_stage(stage_id, model, train_loader, val_loader, criterion, optimizer,
              scheduler, epochs, device, history, best, ckpt_dir):
    """Train for `epochs` epochs, logging + checkpointing each one.

    Input:
        stage_id: 1 or 2 (recorded in the history for plotting).
        history: list of per-epoch dict records, appended in place.
        best: dict {"accuracy": float, "epoch": int} tracking the best model.
        ckpt_dir: directory to save best.pt.

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
        group_lrs = {g.get("name", f"group{i}"): g["lr"]
                     for i, g in enumerate(optimizer.param_groups)}

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, desc, grad_clip=config.GRAD_CLIP)
        val_metrics = evaluate(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        record = {
            "epoch": global_epoch,
            "stage": stage_id,
            # One field per param group: lr_head / lr_encoder / lr_embedding.
            **{f"lr_{name}": v for name, v in group_lrs.items()},
            "time_sec": round(time.time() - t0, 1),
            "timestamp": time.strftime("%m-%d %H:%M:%S"),  # wall clock at epoch END
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)

        lr_str = " ".join(f"{name}={v:.2e}" for name, v in group_lrs.items())
        print(f"[{record['timestamp']}] {desc}  lr[{lr_str}]  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"acc={val_metrics['accuracy']:.4f}  "
              f"macroF1={val_metrics['macro_f1']:.4f}  "
              f"({record['time_sec']}s)")

        # Checkpoint on ACCURACY improvement (the benchmark's metric).
        # Selecting on val loss would be a different -- and with label
        # smoothing, slightly misaligned -- criterion; the two disagree late
        # in training once the model starts overfitting confidence rather
        # than correctness.
        if val_metrics["accuracy"] > best["accuracy"]:
            best["accuracy"] = val_metrics["accuracy"]
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


def collect_run_meta(args, device, train_loader, val_loader, vocab, *,
                     model_name, extra=None):
    """Snapshot the run's config + training params for training_log.json.

    Records enough to reproduce the run from the log alone: corpus/vocab
    stats, model hyperparams, and the two-stage layered-LR schedule.
    """
    meta = {
        "model": model_name,
        "device": str(device),
        "seed": args.seed,
        "data": {
            "data_root": config.DATA_ROOT,
            "train_docs": len(train_loader.dataset),
            "train_batches": len(train_loader),
            "val_docs": len(val_loader.dataset),
            "val_ratio": config.VAL_RATIO,
            "split_seed": config.SPLIT_SEED,
            "batch_size": args.batch_size,
            "eval_batch_size": config.EVAL_BATCH_SIZE,
            "num_workers": args.num_workers,
            "vocab_size": len(vocab),
            "min_freq": config.MIN_FREQ,
            "max_len": config.MAX_LEN,
            "label_counts": train_loader.dataset.label_counts(),
            "val_unk_rate": round(val_loader.dataset.unk_rate(), 4),
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
            name: {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in row.items()}
            for name, row in zip(config.CLASS_NAMES, result["per_class"])
        },
        "matrix": result["matrix"],
    }


def plot_curves(history, output_dir, title_prefix="RNN"):
    """Plot training curves from the history and save PNGs to output_dir.

    Produces:
        loss_curve.png - train vs. val cross-entropy loss per epoch.
        acc_curve.png  - val accuracy and macro-F1 per epoch.
    A dashed vertical line marks the stage-1 -> stage-2 boundary.
    """
    if not history:
        return
    epochs = [r["epoch"] for r in history]
    stage2_start = next((r["epoch"] for r in history if r["stage"] == 2), None)

    # ---- Figure 1: cross-entropy loss, train vs val ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["train_loss"] for r in history], label="train")
    plt.plot(epochs, [r["val_loss"] for r in history], label="val")
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

    # ---- Figure 2: val accuracy + macro-F1 ----
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["accuracy"] for r in history], label="accuracy")
    plt.plot(epochs, [r["macro_f1"] for r in history], label="macro F1")
    if stage2_start is not None:
        plt.axvline(stage2_start - 0.5, color="gray", ls="--", label="stage 2 start")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    # Zoomed y-range: a 4-class model starts at ~0.25 but reaches ~0.9 within
    # one epoch, so plotting from 0 would squash every difference that matters.
    plt.ylim(0.7, 1.0)
    plt.title(f"{title_prefix} val accuracy / macro F1")
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
        torch.optim.Adam with one param_group per ACTIVE tier, head first.
    """
    groups = model.parameter_groups()
    param_groups = []
    for name, lr in (("head", lr_head),
                     ("encoder", lr_encoder),
                     ("embedding", lr_embedding)):
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
    p = argparse.ArgumentParser(description="Train an RNN topic classifier on AG News")
    p.add_argument("--download", action="store_true",
                   help="download AG News (and GloVe) before training")
    p.add_argument("--device", default=config.DEVICE, help="cuda / mps / cpu / auto")
    p.add_argument("--seed", type=int, default=config.SEED,
                   help="training seed. The train/val SPLIT is controlled "
                        "separately by config.SPLIT_SEED and does not move, "
                        "so different seeds are scored on identical data")
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--epochs-stage1", type=int, default=config.STAGE1_EPOCHS,
                   help="stage-1 epochs (frozen embeddings); 0 skips stage 1 "
                        "(use 0 together with --no-glove)")
    p.add_argument("--epochs-stage2", type=int, default=config.STAGE2_EPOCHS,
                   help="stage-2 epochs (embeddings unfrozen); 0 skips stage 2")
    p.add_argument("--cell", choices=["rnn", "lstm", "gru"], default=config.CELL,
                   help="recurrent cell: rnn (vanilla) / lstm / gru. Each "
                        "writes to its own outputs_<cell>/ folder")
    p.add_argument("--pooling", choices=["last", "max", "mean"], default=config.POOLING,
                   help="how token features collapse into a document vector")
    p.add_argument("--no-glove", action="store_true",
                   help="train word vectors from scratch (no GloVe init)")
    p.add_argument("--output-dir", default=None,
                   help="output folder name/path. Default: outputs_<cell>/. A "
                        "bare name is placed under the project root; an "
                        "absolute path is used as-is.")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    use_glove = config.USE_GLOVE and not args.no_glove

    # Output folder: one per cell so the three experiments never overwrite
    # each other; --output-dir overrides (used for ablations).
    if args.output_dir:
        output_dir = (args.output_dir if os.path.isabs(args.output_dir)
                      else os.path.join(config.PROJECT_ROOT, args.output_dir))
    else:
        output_dir = config.output_dir_for_cell(args.cell)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}  seed: {args.seed}")
    print(f"Data root: {config.DATA_ROOT}")
    print(f"Output dir: {output_dir}  ({args.cell} cell, {args.pooling} pooling, "
          f"{'GloVe' if use_glove else 'scratch'} embeddings)")

    # ---- Data ----
    train_loader, val_loader, vocab = build_dataloaders(
        args.batch_size, args.num_workers, args.download, device, use_glove)
    print(f"Train docs: {len(train_loader.dataset)}  ({len(train_loader)} batches)  "
          f"Val docs: {len(val_loader.dataset)}  Vocab: {len(vocab)}")
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

    # Plain cross-entropy: one prediction per document, no positions to
    # ignore, no auxiliary heads -- so there is nothing for a Loss class to
    # wrap. Label smoothing replaces the hard target (1, 0, 0, 0) with
    # (1-eps, eps/3, eps/3, eps/3): being right is still rewarded, being
    # *certain* is not. Class weights are pointless here -- AG News is exactly
    # balanced. nn.CrossEntropyLoss takes RAW logits (log_softmax is applied
    # internally), which is why the model never ends with a softmax.
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
          f"(embedding {len(vocab) * config.EMBED_DIM / 1e6:.2f}M)")

    # From-scratch guard: stage 1 freezes the embedding, which only makes
    # sense when it is pretrained. Warn (don't override) so the run stays
    # reproducible from the command line alone.
    if not model.embedding.pretrained_loaded and args.epochs_stage1 > 0:
        print("[ag-news] WARNING: no GloVe vectors were loaded but stage 1 will "
              "freeze the (random) embedding table. Consider "
              "--epochs-stage1 0 for from-scratch training.")

    history = []
    best = {"accuracy": -1.0, "epoch": -1}
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    # ---- Stage 1: freeze the embedding, train encoder + head ----
    if args.epochs_stage1 > 0:
        print("\n=== Stage 1: freeze embeddings, train encoder + head ===")
        model.freeze_embedding()
        print(f"Trainable params: {count_trainable(model):.2f}M")
        optimizer = build_layered_optimizer(
            model,
            lr_head=config.STAGE1_LR_HEAD,
            lr_encoder=config.STAGE1_LR_ENCODER,
            lr_embedding=None,                     # frozen -> tier skipped
            weight_decay=config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage1)
        best = run_stage(1, model, train_loader, val_loader, criterion, optimizer,
                         scheduler, args.epochs_stage1, device, history, best, output_dir)

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
        best = run_stage(2, model, train_loader, val_loader, criterion, optimizer,
                         scheduler, args.epochs_stage2, device, history, best, output_dir)

    # ---- Save logs + curves ----
    extra = {"glove": use_glove,
             "glove_path": config.GLOVE_PATH if use_glove else None}
    if use_glove:
        extra["glove_coverage"] = round(n_found / len(vocab), 4)
    meta = collect_run_meta(args, device, train_loader, val_loader, vocab,
                            model_name=f"Bi{args.cell.upper()} ({args.pooling} pooling)",
                            extra=extra)
    # Recorded here rather than inside collect_run_meta so "started" is the
    # real start time (the SST-2 project got this wrong and its logs hold the
    # FINISH time under that name).
    meta["started"] = started
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["best_val"] = {"accuracy": round(best["accuracy"], 4), "epoch": best["epoch"]}
    save_log(history, output_dir, meta)
    plot_curves(history, output_dir, title_prefix=f"Bi{args.cell.upper()}")
    print(f"\nDone. Best VAL accuracy={best['accuracy']:.4f} @ epoch {best['epoch']}")
    print(f"Artifacts written to: {output_dir}")

    # ---- Final report on the best checkpoint (per-class + confusion matrix) --
    # Still on VAL, not test: this run must not look at test.csv at all.
    # Run `python eval.py --split test` afterwards for the reported number.
    best_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        print("\nBest checkpoint on the AG News validation split:")
        result = compute_accuracy(model, val_loader, device)  # verbose report
        plot_confusion_matrix(
            result["matrix"], config.CLASS_NAMES,
            os.path.join(output_dir, "confusion_matrix.png"),
            title=f"Bi{args.cell.upper()} val confusion matrix")
        meta["final_val"] = summarize_result(result)
        save_log(history, output_dir, meta)
        print("\nNow score the held-out test split:")
        print(f"  python eval.py --weights {os.path.join(output_dir, 'best.pt')} "
              f"--split test --save-cm")


if __name__ == "__main__":
    main()
    # Must be the last thing that runs: see clean_exit() for why a normal
    # return crashes this process with -1073740791 on Windows.
    clean_exit()
