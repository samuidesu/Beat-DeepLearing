"""Evaluation entry point: score a trained AG News checkpoint on a split.

Prints accuracy / macro-F1, the per-class precision-recall table and the
confusion matrix, and optionally saves the matrix as a PNG.

THIS is where test.csv gets read. train.py deliberately never touches it --
AG News publishes its test labels, so evaluating every epoch on test and then
keeping the best epoch would turn the headline into a best-of-N number. The
honest protocol is: select on --split val, report --split test, once.

Usage:
    python eval.py --split test             # LSTM, outputs_lstm/best.pt
    python eval.py --cell gru --split test  # GRU,  outputs_gru/best.pt
    python eval.py --weights path/to.pt --vocab path/to/vocab.json
    python eval.py --split val              # what training selected on
    python eval.py --split train            # sanity check: fit on train data
    python eval.py --split test --save-cm   # write confusion_matrix_test.png

The model shape (cell / pooling / widths) MUST match how the checkpoint was
trained or load_state_dict fails loudly -- which is the good outcome, unlike
a silent wrong-config evaluation. To make that easy, the shape is read back
from the run's own training_log.json when one sits next to the checkpoint;
explicit flags override, config.py fills whatever is left.
"""

import json
import os
import argparse

import torch
from torch.utils.data import DataLoader

import config
from model.rnn_classifier import RNNClassifier
from dataset.ag_news import AGNewsDataset, build_vocab_from_train, collate_batch
from dataset.vocab import Vocab
from utils.metrics import compute_accuracy
from utils.viz import plot_confusion_matrix
from train import clean_exit, get_device  # reuse the device picker + exit fix


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate an RNN classifier on AG News")
    p.add_argument("--cell", choices=["rnn", "lstm", "gru"], default=None,
                   help="cell the checkpoint was trained with (MUST match; "
                        "default: read from training_log.json, else config.py)")
    p.add_argument("--pooling", choices=["last", "max", "mean"], default=None,
                   help="pooling the checkpoint was trained with (MUST match)")
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in outputs_<cell>/)")
    p.add_argument("--vocab", default=None,
                   help="vocab.json path (default: next to the checkpoint; "
                        "falls back to rebuilding it from train.csv)")
    p.add_argument("--split", choices=["test", "val", "train"], default="test",
                   help="which split to score. test = the reported number; "
                        "val = what training selected on; train = sanity check")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    p.add_argument("--max-batches", type=int, default=None,
                   help="limit the number of batches (quick spot-check)")
    p.add_argument("--save-cm", action="store_true",
                   help="save confusion_matrix_<split>.png next to the checkpoint")
    return p.parse_args()


def read_run_meta(output_dir: str) -> dict:
    """Return the model_cfg block of a run's training_log.json, or {}.

    Lets eval.py and predict.py rebuild the exact architecture a checkpoint
    was trained with instead of trusting that config.py has not been edited
    since -- the failure mode that would cause is a shape error at best and a
    wrong number at worst.
    """
    path = os.path.join(output_dir, "training_log.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(payload, dict):
        return (payload.get("meta") or {}).get("model_cfg", {}) or {}
    return {}


def load_model(weights: str, vocab: Vocab, cell: str = None, pooling: str = None,
               device=None, output_dir: str = None):
    """Rebuild the architecture and load a checkpoint into it.

    Input:
        weights: path to best.pt.
        vocab: the Vocab whose size defines the embedding table.
        cell / pooling: explicit overrides (None -> log -> config.py).
        device: torch device to place the model on.
        output_dir: where to look for training_log.json (default: the
            checkpoint's own folder).

    Output:
        the model in eval mode, plus the resolved cfg dict (for printing).
    """
    output_dir = output_dir or os.path.dirname(os.path.abspath(weights))
    meta = read_run_meta(output_dir)

    def pick(name, explicit, fallback):
        """Precedence: explicit flag > training_log.json > config.py."""
        return explicit if explicit is not None else meta.get(name, fallback)

    cfg = {
        "cell": pick("cell", cell, config.CELL),
        "pooling": pick("pooling", pooling, config.POOLING),
        "embed_dim": pick("embed_dim", None, config.EMBED_DIM),
        "hidden_size": pick("hidden_size", None, config.HIDDEN_SIZE),
        "num_layers": pick("num_layers", None, config.NUM_LAYERS),
        "bidirectional": pick("bidirectional", None, config.BIDIRECTIONAL),
    }

    # pretrained_vectors=None: the checkpoint already holds trained word
    # vectors, so GloVe is not needed (or wanted) at evaluation time.
    model = RNNClassifier(
        vocab_size=len(vocab),
        num_classes=config.NUM_CLASSES,
        embed_dim=cfg["embed_dim"],
        hidden_size=cfg["hidden_size"],
        cell=cfg["cell"],
        num_layers=cfg["num_layers"],
        bidirectional=cfg["bidirectional"],
        pooling=cfg["pooling"],
        dropout=config.DROPOUT,          # inactive in eval() mode anyway
        pad_idx=config.PAD_IDX,
        pretrained_vectors=None,
    )
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return (model.to(device) if device is not None else model), cfg


def load_vocab(path: str = None, output_dir: str = None) -> Vocab:
    """Load vocab.json, falling back to rebuilding it from train.csv.

    The rebuild is deterministic (same csv + same config.SPLIT_SEED +
    MIN_FREQ -> same itos), so it recovers a lost vocab.json -- but only as
    long as config.py has not changed since training, hence the warning.
    """
    if path is None and output_dir is not None:
        cand = os.path.join(output_dir, "vocab.json")
        path = cand if os.path.isfile(cand) else None
    if path and os.path.isfile(path):
        return Vocab.load(path)
    print("[eval] vocab.json not found -- rebuilding from train.csv (ids match "
          "only if config.MIN_FREQ / MAX_VOCAB_SIZE / SPLIT_SEED are unchanged)")
    return build_vocab_from_train()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    output_dir = config.output_dir_for_cell(args.cell or config.CELL)
    if args.weights is None:
        args.weights = os.path.join(output_dir, "best.pt")
    else:
        output_dir = os.path.dirname(os.path.abspath(args.weights))

    vocab = load_vocab(args.vocab, output_dir)
    model, cfg = load_model(args.weights, vocab, args.cell, args.pooling,
                            device, output_dir)
    print(f"Loaded weights: {args.weights}")
    print(f"Model: Bi{cfg['cell'].upper()} h={cfg['hidden_size']} "
          f"layers={cfg['num_layers']} pooling={cfg['pooling']}  vocab={len(vocab)}")

    dataset = AGNewsDataset(args.split, vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_batch)
    print(f"{args.split} documents: {len(dataset)}  "
          f"<unk> rate: {dataset.unk_rate():.4f}")

    result = compute_accuracy(model, loader, device, max_batches=args.max_batches)

    if args.save_cm:
        path = os.path.join(output_dir, f"confusion_matrix_{args.split}.png")
        plot_confusion_matrix(
            result["matrix"], config.CLASS_NAMES, path,
            title=f"Bi{cfg['cell'].upper()} {args.split} confusion matrix")
        print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
    clean_exit()   # see train.clean_exit(): cuDNN RNN + CUDA cannot shut down cleanly
