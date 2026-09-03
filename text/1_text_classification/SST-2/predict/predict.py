"""Inference entry point: run a trained classifier on sentences and show the
verdict with its confidence (the text counterpart of the detection projects'
detect.py and the segmentation projects' segment.py -- print a probability
instead of drawing boxes or painting pixels).

This script lives in SST-2/predict/. Results are written to
SST-2/predict/results/predictions.txt, which is wiped and rewritten on every
run.

Four ways to feed it text:
    --text "..."        one or more sentences straight from the command line
    --file sentences.txt one sentence per line
    --dev-random N      N random SST-2 dev sentences (gold labels shown)
    --dev-mistakes N    the N dev sentences the model gets WRONG, most
                        confidently-wrong first -- the single most useful view
                        for figuring out what the model still cannot read

Usage (run from the SST-2 project root):
    python predict/predict.py --text "a charming, funny film"
    python predict/predict.py --cell gru --dev-random 20
    python predict/predict.py --dev-mistakes 15
    python predict/predict.py --file my_reviews.txt
"""

import os
import sys
import random
import shutil
import argparse

import torch
import torch.nn.functional as F

# This file sits in SST-2/predict/, so the project root is its parent's
# parent. Put it on sys.path so `import config`, `model`, `utils`, `eval` ...
# resolve regardless of the current working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config  # noqa: E402
from dataset.sst2 import read_split  # noqa: E402
from dataset.vocab import tokenize  # noqa: E402
from utils.viz import format_prediction  # noqa: E402
from train import get_device  # noqa: E402  reuse the device picker
from eval import load_model, load_vocab  # noqa: E402  reuse the checkpoint loader

# Default output folder: SST-2/predict/results/ (sits next to this file).
_PREDICT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_PREDICT_DIR, "results")


def parse_args():
    p = argparse.ArgumentParser(description="Classify sentences with a trained SST-2 model")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", nargs="+", help="one or more sentences to classify")
    src.add_argument("--file", help="a text file with one sentence per line")
    src.add_argument("--dev-random", type=int, metavar="N",
                     help="randomly sample N sentences from the SST-2 dev split")
    src.add_argument("--dev-mistakes", type=int, metavar="N",
                     help="show the N most confidently WRONG dev predictions")
    p.add_argument("--seed", type=int, default=None,
                   help="random seed for reproducible --dev-random sampling")
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
    """Turn the CLI source flag into a list of (sentence, gold_label_or_None).

    Output:
        list of (str, int | None). Gold labels exist only for the dev-split
        modes; free-form text has none, and format_prediction then prints no
        hit/miss marker.
    """
    if args.text:
        return [(t, None) for t in args.text]

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            return [(ln.strip(), None) for ln in f if ln.strip()]

    # Both dev modes read the labeled dev split.
    pairs = read_split("dev")
    if args.dev_random is not None:
        rng = random.Random(args.seed)
        n = min(args.dev_random, len(pairs))
        return rng.sample(pairs, n)
    return pairs  # --dev-mistakes: score everything, filter after inference


@torch.no_grad()
def predict(model, vocab, sentences, device, batch_size=128):
    """Classify a list of raw sentences.

    Batches them exactly like the training pipeline (tokenize -> encode ->
    pad -> lengths), because ANY difference here is a train/serve skew: the
    same text must produce the same ids it would have produced during
    training.

    Input:
        model: an eval-mode RNNClassifier.
        vocab: the Vocab the checkpoint was trained with.
        sentences: list of raw strings.
        device / batch_size: as usual.
    Output:
        probs [N, C] float tensor of per-class probabilities (on CPU).
    """
    model.eval()
    out = []
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start + batch_size]
        encoded = [vocab.encode(tokenize(s), config.MAX_LEN) or [config.UNK_IDX]
                   for s in chunk]
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
    sentences = [s for s, _ in pairs]
    golds = [y for _, y in pairs]
    probs = predict(model, vocab, sentences, device, args.batch_size)
    preds = probs.argmax(dim=1).tolist()

    if args.dev_mistakes is not None:
        # Keep only the errors, hardest first: sort by the probability the
        # model gave its (wrong) answer, descending. A confidently wrong
        # prediction is a real modeling failure; a 0.51 miss is noise.
        wrong = [i for i, (p, y) in enumerate(zip(preds, golds)) if y is not None and p != y]
        wrong.sort(key=lambda i: float(probs[i, preds[i]]), reverse=True)
        keep = wrong[:args.dev_mistakes]
        print(f"{len(wrong)} wrong out of {len(preds)} dev sentences "
              f"(accuracy {1 - len(wrong) / max(len(preds), 1):.4f}); "
              f"showing the {len(keep)} most confident errors\n")
        pairs = [pairs[i] for i in keep]
        sentences = [sentences[i] for i in keep]
        golds = [golds[i] for i in keep]
        probs = probs[keep] if keep else probs[:0]

    # Fresh results folder every run (same convention as detect/segment).
    out_dir = args.out
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    lines = []
    for sentence, gold, prob in zip(sentences, golds, probs):
        line = format_prediction(sentence, prob, config.CLASS_NAMES, gold=gold)
        print(line)
        lines.append(line)

    # Report accuracy whenever gold labels were available.
    scored = [(int(p.argmax()), y) for p, y in zip(probs, golds) if y is not None]
    summary = ""
    if scored and args.dev_mistakes is None:
        acc = sum(1 for p, y in scored if p == y) / len(scored)
        summary = f"\naccuracy on these {len(scored)} labeled sentences: {acc:.4f}"
        print(summary)

    out_path = os.path.join(out_dir, "predictions.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# weights: {weights}\n# model: Bi{cfg['cell'].upper()} "
                f"pooling={cfg['pooling']}\n")
        f.write("\n".join(lines) + summary + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
