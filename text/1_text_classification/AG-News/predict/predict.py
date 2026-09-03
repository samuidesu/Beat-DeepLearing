"""Inference entry point: run a trained classifier on news text and show the
predicted topic with its confidence (the text counterpart of the detection
projects' detect.py and the segmentation projects' segment.py -- print a
probability instead of drawing boxes or painting pixels).

This script lives in AG-News/predict/. Results are written to
AG-News/predict/results/predictions.txt, which is wiped and rewritten on every
run.

Four ways to feed it text:
    --text "..."         one or more documents straight from the command line
    --file articles.txt  one document per line
    --test-random N      N random test documents (gold labels shown)
    --test-mistakes N    the N test documents the model gets WRONG, most
                         confidently-wrong first -- the single most useful
                         view for figuring out what the model cannot read

Usage (run from the AG-News project root):
    python predict/predict.py --text "Arsenal beat Chelsea 2-1 at the Emirates"
    python predict/predict.py --cell gru --test-random 20
    python predict/predict.py --test-mistakes 15
    python predict/predict.py --file my_headlines.txt
"""

import os
import sys
import random
import shutil
import argparse

import torch
import torch.nn.functional as F

# This file sits in AG-News/predict/, so the project root is its parent's
# parent. Put it on sys.path so `import config`, `model`, `utils`, `eval` ...
# resolve regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from dataset.ag_news import read_split, _clean  # noqa: E402
from dataset.vocab import tokenize  # noqa: E402
from utils.viz import format_prediction  # noqa: E402
from train import clean_exit, get_device  # noqa: E402  device picker + exit fix
from eval import load_model, load_vocab  # noqa: E402  reuse the checkpoint loader

# Default output folder: AG-News/predict/results/ (sits next to this file).
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def parse_args():
    p = argparse.ArgumentParser(description="Classify news text with a trained AG News model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+", help="one or more documents to classify")
    src.add_argument("--file", help="a text file with one document per line")
    src.add_argument("--test-random", type=int, metavar="N",
                     help="randomly sample N documents from the test split")
    src.add_argument("--test-mistakes", type=int, metavar="N",
                     help="show the N most confidently WRONG test predictions")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible --test-random sampling")
    p.add_argument("--cell", choices=["rnn", "lstm", "gru"], default=None,
                   help="cell the checkpoint was trained with (default: read "
                        "from training_log.json, else config.py)")
    p.add_argument("--pooling", choices=["last", "max", "mean"], default=None)
    p.add_argument("--weights", default=None,
                   help="checkpoint path (default: best.pt in outputs_<cell>/)")
    p.add_argument("--vocab", default=None, help="vocab.json path")
    p.add_argument("--out", default=_RESULTS_DIR,
                   help="output folder (wiped and recreated fresh each run)")
    p.add_argument("--device", default=config.DEVICE)
    p.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    return p.parse_args()


def collect_inputs(args):
    """Turn the CLI source flag into a list of (text, gold_label_or_None).

    Output:
        list of (str, int | None). Gold labels exist only for the test-split
        modes; free-form text has none, and format_prediction then prints no
        hit/miss marker.
    """
    if args.text:
        return [(t, None) for t in args.text]

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            return [(ln.strip(), None) for ln in f if ln.strip()]

    # Both test modes read the labeled test split.
    pairs = read_split("test")
    if args.test_random is not None:
        rng = random.Random(args.seed)
        return rng.sample(pairs, min(args.test_random, len(pairs)))
    return pairs  # --test-mistakes: score everything, filter after inference


@torch.no_grad()
def predict(model, vocab, texts, device, batch_size=256):
    """Classify a list of raw documents.

    Batches them exactly like the training pipeline (clean -> tokenize ->
    encode -> pad -> lengths), because ANY difference here is train/serve
    skew: the same text must produce the same ids it would have produced
    during training. _clean() in particular is easy to forget -- read_split()
    applies it for the test modes, but text arriving via --text or --file has
    not been through it, so it is applied here for every path.

    Input:
        model: an eval-mode RNNClassifier.
        vocab: the Vocab the checkpoint was trained with.
        texts: list of raw strings.
        device / batch_size: as usual.
    Output:
        probs [N, C] float tensor of per-class probabilities (on CPU).
    """
    model.eval()
    out = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        encoded = [vocab.encode(tokenize(_clean(t)), config.MAX_LEN) or [config.UNK_IDX]
                   for t in chunk]
        lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
        ids = torch.full((len(encoded), int(lengths.max())), config.PAD_IDX,
                         dtype=torch.long)
        for i, e in enumerate(encoded):
            ids[i, :len(e)] = torch.tensor(e, dtype=torch.long)

        logits = model(ids.to(device), lengths.to(device))
        # softmax turns raw logits into the probabilities we print. The model
        # never applies it internally -- cross-entropy wants raw logits.
        out.append(F.softmax(logits, dim=1).cpu())
    return torch.cat(out) if out else torch.empty(0, config.NUM_CLASSES)


def main():
    args = parse_args()
    device = get_device(args.device)

    output_dir = config.output_dir_for_cell(args.cell or config.CELL)
    weights = args.weights or os.path.join(output_dir, "best.pt")
    if args.weights:
        output_dir = os.path.dirname(os.path.abspath(args.weights))
    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"{weights} not found -- train a model first (python train.py)")

    vocab = load_vocab(args.vocab, output_dir)
    model, cfg = load_model(weights, vocab, args.cell, args.pooling, device, output_dir)
    print(f"Device: {device}")
    print(f"Loaded weights: {weights}")
    print(f"Model: Bi{cfg['cell'].upper()} pooling={cfg['pooling']}  vocab={len(vocab)}\n")

    pairs = collect_inputs(args)
    texts = [t for t, _ in pairs]
    golds = [y for _, y in pairs]
    probs = predict(model, vocab, texts, device, args.batch_size)
    preds = probs.argmax(dim=1).tolist()

    if args.test_mistakes is not None:
        # Keep only the errors, hardest first: sort by the probability the
        # model gave its (wrong) answer, descending. A confidently wrong
        # prediction is a real modeling failure; a 0.26 miss between two
        # plausible topics is noise.
        wrong = [i for i, (p, y) in enumerate(zip(preds, golds))
                 if y is not None and p != y]
        wrong.sort(key=lambda i: float(probs[i, preds[i]]), reverse=True)
        keep = wrong[:args.test_mistakes]
        print(f"{len(wrong)} wrong out of {len(preds)} test documents "
              f"(accuracy {1 - len(wrong) / max(len(preds), 1):.4f}); "
              f"showing the {len(keep)} most confident errors\n")
        texts = [texts[i] for i in keep]
        golds = [golds[i] for i in keep]
        probs = probs[keep] if keep else probs[:0]

    # Fresh results folder every run (same convention as detect/segment).
    out_dir = args.out
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    for text, gold, prob in zip(texts, golds, probs):
        line = format_prediction(text, prob, config.CLASS_NAMES, gold=gold)
        print(line)
        lines.append(line)

    # Report accuracy whenever gold labels were available.
    scored = [(int(p.argmax()), y) for p, y in zip(probs, golds) if y is not None]
    summary = ""
    if scored and args.test_mistakes is None:
        acc = sum(1 for p, y in scored if p == y) / len(scored)
        summary = f"\naccuracy on these {len(scored)} labeled documents: {acc:.4f}"
        print(summary)

    out_path = os.path.join(out_dir, "predictions.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# weights: {weights}\n# model: Bi{cfg['cell'].upper()} "
                f"pooling={cfg['pooling']}\n")
        f.write("\n".join(lines) + summary + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
    clean_exit()   # see train.clean_exit(): cuDNN RNN + CUDA cannot shut down cleanly
