```python
from pathlib import Path
import importlib.util
import re


def patch_ultralytics_focal_loss():
    """
    Patch Ultralytics YOLOv8 v8DetectionLoss:
    replace BCE classification loss with Focal Loss modulation.

    Purpose:
    - handle class imbalance
    - focus more on hard samples
    """

    # 1. 找到当前 Python 环境中的 ultralytics/utils/loss.py
    spec = importlib.util.find_spec("ultralytics.utils.loss")

    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Cannot find ultralytics.utils.loss. "
            "Please check whether ultralytics is installed."
        )

    loss_path = Path(spec.origin)
    text = loss_path.read_text(encoding="utf-8")

    marker = "FDM_FOCAL_LOSS_PATCH"

    # 2. 防止重复 patch
    if marker in text:
        print(f"[FocalLoss] Already patched: {loss_path}")
        return

    # 3. 找到 YOLOv8 原来的分类 BCE Loss 代码
    pattern = (
        r"^(\s*)loss\[1\]\s*=\s*self\.bce\(pred_scores,\s*target_scores\.to\(dtype\)\)"
        r"\.sum\(\)\s*/\s*target_scores_sum.*$"
    )

    match = re.search(pattern, text, flags=re.MULTILINE)

    if match is None:
        raise RuntimeError(
            "Cannot find YOLOv8 classification BCE loss line. "
            "Please open ultralytics/utils/loss.py and search for loss[1] manually."
        )

    # 4. 获取原代码的缩进，保证替换进去的代码缩进正确
    indent = match.group(1)

    # 5. Focal Loss 替换代码
    replacement_body = f"""# {marker}: replace BCE cls loss with Focal Loss
target_cls = target_scores.to(dtype)

# Original BCE classification loss
cls_loss = self.bce(pred_scores, target_cls)

# Focal Loss parameters
gamma = 1.5
alpha = 0.25

# Convert logits to probabilities
pred_prob = pred_scores.sigmoid()

# p_t means the probability assigned to the true class
p_t = target_cls * pred_prob + (1 - target_cls) * (1 - pred_prob)

# alpha balances positive and negative samples
alpha_factor = target_cls * alpha + (1 - target_cls) * (1 - alpha)

# gamma down-weights easy samples and focuses on hard samples
modulating_factor = (1.0 - p_t) ** gamma

# Apply Focal Loss modulation to BCE classification loss
cls_loss = cls_loss * alpha_factor * modulating_factor

# Classification loss
loss[1] = cls_loss.sum() / target_scores_sum  # Focal cls loss"""

    # 给 replacement_body 每一行都加上原来的缩进
    replacement = "\n".join(
        indent + line if line.strip() else line
        for line in replacement_body.splitlines()
    )

    # 6. 替换原来的 BCE 分类损失
    new_text = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)

    # 7. 写回 ultralytics/utils/loss.py
    loss_path.write_text(new_text, encoding="utf-8")

    print(f"[FocalLoss] YOLOv8 classification loss patched successfully: {loss_path}")


if __name__ == "__main__":
    patch_ultralytics_focal_loss()
```
