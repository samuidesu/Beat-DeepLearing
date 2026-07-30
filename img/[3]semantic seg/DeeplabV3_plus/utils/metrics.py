"""Evaluation metric: mean Intersection-over-Union (mIoU) + pixel accuracies.

Detection's mAP was heavy machinery (decode, NMS, PR curves -- we delegated it
to torchmetrics). Segmentation's standard metric is simple enough to build
from scratch, and building it shows exactly what it measures:

ONE confusion matrix M [K, K] accumulated over ALL pixels of ALL images,
where M[g, p] = number of pixels whose GT class is g and predicted class is p.
From it, for each class c:

    TP = M[c, c]                     (pixels correctly labeled c)
    FP = column_sum(c) - TP          (pixels wrongly labeled c)
    FN = row_sum(c) - TP             (class-c pixels labeled something else)
    IoU_c = TP / (TP + FP + FN)

    mIoU      = mean of IoU_c over classes         <- the headline number
    pixel_acc = trace(M) / sum(M)                  (dominated by background!)
    mean_acc  = mean over classes of TP / row_sum  (per-class recall)

mIoU treats every CLASS equally regardless of pixel count -- that's why it is
the standard: pixel_acc can look great (~90%) while small classes are garbage,
because background alone is ~70% of pixels.

Ignored pixels (GT == 255: VOC void contours and our padding) are dropped
BEFORE counting, so they influence nothing. Note the accumulate-then-compute
order: sum the matrix over the whole dataset FIRST, divide LAST -- averaging
per-image IoUs instead would over-weight images with tiny class regions.
"""

import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
import config  # noqa: E402


class ConfusionMatrix:
    """Streaming K x K pixel confusion matrix.

    Usage: create once, .update() per batch, .compute() at the end.
    Kept on CPU: per batch it only receives two flattened id tensors, and CPU
    bincount sidesteps device quirks (e.g. MPS op gaps).
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        # mat[g, p] = pixel count with GT class g, predicted class p.
        self.mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def reset(self):
        """Zero the matrix (start a fresh evaluation)."""
        self.mat.zero_()

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        """Accumulate one batch of predictions.

        Input:
            pred: predicted class ids, any shape (e.g. [B, H, W]), ints 0..K-1
                  (i.e. logits.argmax(dim=1) -- NOT raw logits).
            gt:   ground-truth ids, same shape; values outside [0, K) -- i.e.
                  the 255 ignore label -- are dropped here.
        """
        pred = pred.flatten().cpu()
        gt = gt.flatten().cpu()
        keep = (gt >= 0) & (gt < self.num_classes)   # drops ignore=255
        # Encode each (gt, pred) pair as one integer gt*K + pred, histogram
        # them all at once with bincount, then reshape back to K x K.
        idx = gt[keep] * self.num_classes + pred[keep]
        counts = torch.bincount(idx, minlength=self.num_classes ** 2)
        self.mat += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        """Reduce the matrix to metrics.

        Output dict:
            miou:       float, mean IoU over classes PRESENT in GT or pred
                        (on full VOC val all 21 classes appear).
            pixel_acc:  float, overall fraction of correctly labeled pixels.
            mean_acc:   float, mean per-class recall.
            per_class_iou: list of `num_classes` floats; float("nan") for a
                        class absent from both GT and predictions.
        """
        mat = self.mat.double()
        tp = mat.diag()                       # [K]
        gt_count = mat.sum(dim=1)             # row sums: GT pixels per class
        pred_count = mat.sum(dim=0)           # col sums: predicted pixels per class
        union = gt_count + pred_count - tp    # TP+FP+FN

        present = union > 0                   # class appears in GT or pred
        iou = torch.full_like(tp, float("nan"))
        iou[present] = tp[present] / union[present]

        acc_present = gt_count > 0
        acc = tp[acc_present] / gt_count[acc_present]

        total = mat.sum().clamp(min=1)
        return {
            "miou": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_acc": float(tp.sum() / total),
            "mean_acc": float(acc.mean()) if acc_present.any() else 0.0,
            "per_class_iou": [float(v) for v in iou],
        }


def print_per_class(result: dict, class_names=None):
    """Print a per-class IoU table, WORST class first (diagnosis order).

    Same idea as the FCOS project's eval_per_class.py -- but here the
    per-class numbers come free from the confusion matrix; no extra pass or
    matching logic is needed.

    Input:
        result: the dict from ConfusionMatrix.compute() / compute_miou().
        class_names: list of names indexed by class id (default VOC's 21).
    """
    names = class_names or config.VOC_SEG_CLASSES
    rows = [(name, iou) for name, iou in zip(names, result["per_class_iou"])]
    rows.sort(key=lambda r: (r[1] != r[1], r[1]))  # NaN last, then ascending
    print(f"\n{'class':<14} {'IoU':>7}")
    print("-" * 22)
    for name, iou in rows:
        iou_s = "  n/a" if iou != iou else f"{iou:7.4f}"
        print(f"{name:<14} {iou_s}")
    print("-" * 22)
    print(f"{'mIoU':<14} {result['miou']:7.4f}")
    print(f"{'pixel_acc':<14} {result['pixel_acc']:7.4f}")
    print(f"{'mean_acc':<14} {result['mean_acc']:7.4f}")


@torch.no_grad()
def compute_miou(model, loader, device, num_classes: int = None,
                 max_batches=None, verbose: bool = True) -> dict:
    """Run the model over `loader` and compute mIoU (& friends).

    The segmentation counterpart of the detection projects' compute_map --
    same call shape, so train.py/eval.py stay parallel.

    Input:
        model:  segmentation model returning logits [B, C, H, W]
                (H, W = input size).
        loader: yields (images, masks); the val loader uses batch_size=1.
        device: torch device.
        num_classes: defaults to config.NUM_CLASSES (21).
        max_batches: if set, only evaluate this many batches (quick proxy).
        verbose: print the summary + per-class table.

    Output:
        the ConfusionMatrix.compute() dict (miou / pixel_acc / mean_acc /
        per_class_iou).
    """
    num_classes = num_classes or config.NUM_CLASSES
    model.eval()
    cm = ConfusionMatrix(num_classes)

    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        logits = model(images)                # [B, C, H, W]
        # argmax over the class dim -> hard per-pixel prediction. masks stay
        # on CPU (update() moves everything to CPU anyway).
        cm.update(logits.argmax(dim=1), masks)

    result = cm.compute()
    if verbose:
        print_per_class(result)
    return result


# ---- Quick self-test: run this file directly (hand-checkable numbers) --------
# python utils/metrics.py
if __name__ == "__main__":
    # 3 classes, one 4x4 "image", built so the answer is computable by hand.
    #   GT:   class 0 on the top two rows, class 1 on the bottom two,
    #         except one ignored (255) pixel.
    #   Pred: perfect, except two class-0 pixels predicted as 2,
    #         and the ignored pixel predicted "wrong" (must not count).
    gt = torch.tensor([[0, 0, 0, 0],
                       [0, 0, 0, 255],
                       [1, 1, 1, 1],
                       [1, 1, 1, 1]])
    pred = torch.tensor([[0, 0, 2, 2],
                         [0, 0, 0, 1],
                         [1, 1, 1, 1],
                         [1, 1, 1, 1]])

    cm = ConfusionMatrix(num_classes=3)
    cm.update(pred, gt)
    res = cm.compute()

    # By hand: class0 TP=5, FN=2, FP=0 -> IoU 5/7 ; class1 TP=8, FP=0 -> 1.0 ;
    # class2 TP=0, FP=2 -> 0.0 ; mIoU = (5/7 + 1 + 0)/3 ≈ 0.5714.
    print("per-class IoU:", [round(v, 4) for v in res["per_class_iou"]],
          "(expected [0.7143, 1.0, 0.0])")
    print("mIoU:", round(res["miou"], 4), "(expected 0.5714)")
    print("pixel_acc:", round(res["pixel_acc"], 4), "(expected 13/15 = 0.8667)")
