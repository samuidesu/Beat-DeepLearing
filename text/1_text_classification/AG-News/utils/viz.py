"""Visualization for text classification: readable predictions + a heatmap.

Segmentation could paint its output straight onto the image; a topic
prediction has no picture, so "visualization" here means two much smaller
things that are still the fastest way to see what the model is doing:

  1. format_prediction() -- one console line per document with the predicted
     class, a confidence bar, and (when the gold label is known) a hit/miss
     marker. Reading 20 of these tells you more than the accuracy number does:
     you see whether the mistakes are near-ties between two plausible topics
     or confident and wrong.
  2. plot_confusion_matrix() -- the 4x4 counts as a labeled heatmap. With four
     classes this earns its keep in a way it did not on binary SST-2: the
     interesting question is not "how many errors" but "WHICH PAIR of topics",
     and that is a shape the eye reads off a heatmap instantly.

Note the difference from the SST-2 project's version of this file: there, the
probability bar was centered on 0.5 because the decision boundary of a binary
classifier IS 0.5 and the direction of the bar carried the answer. With four
classes there is no single boundary (the winner can take as little as 0.26),
so the bar simply reports the winning class's confidence and the class name
carries the answer.
"""

import os

import numpy as np


def prob_bar(p: float, width: int = 20) -> str:
    """Render a probability in [0, 1] as a fixed-width text bar."""
    p = min(max(float(p), 0.0), 1.0)
    filled = round(p * width)
    return "#" * filled + "-" * (width - filled)


def format_prediction(text: str, probs, class_names, gold: int = None,
                      max_chars: int = 70) -> str:
    """One-line summary of a single prediction.

    Input:
        text: the raw document.
        probs: per-class probabilities (list/tensor of length C).
        class_names: e.g. ["World", "Sports", "Business", "Sci/Tech"].
        gold: true class id, or None when unknown (free-form user text).
        max_chars: truncate the text so the columns stay aligned.
    Output:
        a formatted string, e.g.
        "[Sports   0.97] ###################- OK   Arsenal loses 100 per ..."
        On a miss the gold class is appended, since with 4 classes "wrong" is
        not self-explanatory the way it is with 2.
    """
    probs = [float(p) for p in probs]
    pred = max(range(len(probs)), key=probs.__getitem__)

    mark = "    "
    if gold is not None:
        mark = "OK  " if pred == gold else "MISS"

    text = " ".join(text.split())          # collapse whitespace for one line
    shown = text if len(text) <= max_chars else text[:max_chars - 3] + "..."
    line = (f"[{class_names[pred]:<8} {probs[pred]:.2f}] "
            f"{prob_bar(probs[pred])} {mark} {shown}")
    if gold is not None and pred != gold:
        line += f"   (gold: {class_names[gold]})"
    return line


def plot_confusion_matrix(matrix, class_names, path: str, title: str = None,
                          normalize: bool = True):
    """Save the confusion matrix as an annotated heatmap PNG.

    Input:
        matrix: K x K nested list / array of counts (rows = true class).
        class_names: labels for both axes.
        path: output .png path.
        title: figure title (defaults to "confusion matrix").
        normalize: color cells by ROW-normalized rate (per-class recall) while
            still printing the raw counts. Without it a class with more
            samples simply looks darker, which says nothing about quality.
            (AG News is exactly balanced, so here the two colorings would look
            almost identical -- the flag stays for consistency with the rest
            of the repo, where it matters.)
    """
    import matplotlib
    matplotlib.use("Agg")               # headless: no display needed
    import matplotlib.pyplot as plt

    counts = np.asarray(matrix, dtype=np.float64)
    shown = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None) if normalize else counts

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=shown.max())
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or "confusion matrix")

    # Annotate every cell with the raw count (and the rate when normalizing),
    # flipping the text color on dark cells so it stays readable.
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            label = (f"{int(counts[i, j])}\n{shown[i, j]:.1%}"
                     if normalize else f"{int(counts[i, j])}")
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > shown.max() * 0.6 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- Quick self-test: run this file directly --------------------------------
# python utils/viz.py
if __name__ == "__main__":
    names = ["World", "Sports", "Business", "Sci/Tech"]
    print(format_prediction("Arsenal loses 100 per cent record with late Bolton equaliser",
                            [0.01, 0.97, 0.01, 0.01], names, gold=1))
    print(format_prediction("Oil prices climb as OPEC holds output steady",
                            [0.05, 0.00, 0.88, 0.07], names, gold=2))
    print(format_prediction("Intel posts record quarterly earnings",
                            [0.02, 0.01, 0.55, 0.42], names, gold=3))
    print(format_prediction("some unlabeled user text about a new phone",
                            [0.10, 0.05, 0.25, 0.60], names))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cm_selftest.png")
    plot_confusion_matrix([[1700, 60, 90, 50], [40, 1830, 20, 10],
                           [80, 15, 1600, 205], [45, 10, 190, 1655]], names, out,
                          title="self-test")
    print(f"\nwrote {out} ({os.path.getsize(out)} bytes) -- delete it when done")
