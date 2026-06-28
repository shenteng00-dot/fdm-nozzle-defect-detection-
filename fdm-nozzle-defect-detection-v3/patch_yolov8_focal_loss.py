from pathlib import Path
import importlib.util
import re


def patch_ultralytics_focal_loss():
    """
    Patch Ultralytics YOLOv8 v8DetectionLoss:
    replace BCE classification loss with Focal Loss modulation.

    This is used for class imbalance and hard-sample learning.
    """

    spec = importlib.util.find_spec("ultralytics.utils.loss")

    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot find ultralytics.utils.loss. Please check whether ultralytics is installed.")

    loss_path = Path(spec.origin)
    text = loss_path.read_text(encoding="utf-8")

    marker = "FDM_FOCAL_LOSS_PATCH"

    if marker in text:
        print(f"[FocalLoss] Already patched: {loss_path}")
        return

    pattern = r"^(\s*)loss\[1\]\s*=\s*self\.bce\(pred_scores,\s*target_scores\.to\(dtype\)\)\.sum\(\)\s*/\s*target_scores_sum.*$"

    match = re.search(pattern, text, flags=re.MULTILINE)

    if match is None:
        raise RuntimeError(
            "Cannot find YOLOv8 classification BCE loss line. "
            "Please open ultralytics/utils/loss.py and search for loss[1] manually."
        )

    indent = match.group(1)

    replacement = f"""{indent}# {marker}: replace BCE cls loss with Focal Loss
{indent}target_cls = target_scores.to(dtype)
{indent}cls_loss = self.bce(pred_scores, target_cls)

{indent}# Focal Loss parameters
{indent}gamma = 1.5
{indent}alpha = 0.25

{indent}# Convert logits to probabilities
{indent}pred_prob = pred_scores.sigmoid()

{indent}# p_t means the probability assigned to the true class
{indent}p_t = target_cls * pred_prob + (1 - target_cls) * (1 - pred_prob)

{indent}# alpha balances positive and negative samples
{indent}alpha_factor = target_cls * alpha + (1 - target_cls) * (1 - alpha)

{indent}# gamma down-weights easy samples and focuses on hard samples
{indent}modulating_factor = (1.0 - p_t) ** gamma

{indent}# Apply Focal Loss modulation to BCE classification loss
{indent}cls_loss = cls_loss * alpha_factor * modulating_factor

{indent}loss[1] = cls_loss.sum() / target_scores_sum  # Focal cls loss"""

    new_text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)

    loss_path.write_text(new_text, encoding="utf-8")

    print(f"[FocalLoss] YOLOv8 classification loss patched successfully: {loss_path}")
