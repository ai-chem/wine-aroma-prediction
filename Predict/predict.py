import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from contextlib import nullcontext

from torch.serialization import add_safe_globals
import torch.torch_version as _tv
add_safe_globals([_tv.TorchVersion])

DEFAULT_LABEL_NAMES = [
    "Fruits",
    "Berries",
    "Flowers",
    "Herbs/Spices",
    "Wood",
    "Tobacco/Smoke",
    "Citrus",
    "Nuts",
    "Coffee",
    "Chocolate/Cacao",
]

def _resolve_label_names(n_classes: int, ckpt_names):
    """
    Pick label names from checkpoint if available; otherwise use defaults.
    If count mismatches, fallback to generic class_i.
    """
    # checkpoint-supplied names
    if isinstance(ckpt_names, (list, tuple)) and len(ckpt_names) == n_classes:
        return list(map(str, ckpt_names)), True

    # fallback to defaults (only if sizes match)
    if len(DEFAULT_LABEL_NAMES) == n_classes:
        return DEFAULT_LABEL_NAMES[:], True

    # final fallback: generic class_i
    print(f"[WARN] n_classes={n_classes} but have {len(ckpt_names) if ckpt_names is not None else 'None'} ckpt names "
          f"and {len(DEFAULT_LABEL_NAMES)} default names; using generic class_i.")
    return [f"class_{i}" for i in range(n_classes)], False

class ConvNet(nn.Module):
    def __init__(self, depth=2, n_classes=10, p=0.2, widths=(64,128,256,512,512)):
        super().__init__()
        self.depth = depth
        layers, in_ch = [], 1
        for i in range(depth):
            out_ch = widths[i]
            layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.fc1 = nn.Linear(in_ch, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_classes)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x)); x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)  # logits

def _to_chw1(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[None, ...]
    elif not (x.ndim == 3 and x.shape[0] == 1):
        raise ValueError(f"Expected [H,W] or [1,H,W], got {x.shape}")
    x = x.astype(np.float32)
    m, s = x.mean(), x.std()
    if s == 0: s = 1.0
    return (x - m) / (s + 1e-6)

def _find_best_ckpt(weights_dir: Path) -> Path:
    cands = list(weights_dir.rglob("*.pt"))
    if not cands:
        raise FileNotFoundError(f"No .pt files under {weights_dir}")
    have_any_metrics = False
    scored = []
    for p in cands:
        try:
            ckpt = torch.load(p, map_location="cpu")
            m = ckpt.get("metrics", {}) or {}
            vl = float(m.get("val_loss", math.inf))
            va = float(m.get("val_acc", -math.inf))
            if math.isfinite(vl) or math.isfinite(va):
                have_any_metrics = True
        except Exception:
            vl, va = math.inf, -math.inf
        mt = p.stat().st_mtime
        scored.append((p, vl, va, mt))
    if have_any_metrics:
        scored.sort(key=lambda t: (t[1], -t[2], -t[3]))
        return scored[0][0]
    else:
        return max(cands, key=lambda p: p.stat().st_mtime)

@torch.no_grad()
def _predict_array(model: nn.Module, device: str, arr: np.ndarray, threshold: float):
    x = torch.from_numpy(_to_chw1(arr))[None].to(device)
    use_amp = (device == "cuda")
    ctx = torch.amp.autocast(device_type="cuda", enabled=use_amp) if use_amp else nullcontext()
    with ctx:
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    preds = (probs >= threshold).astype(int)
    return probs, preds

def _load_model_and_meta(ckpt_path: Path, device: str, depth_arg: int = None, n_classes_arg: int = None):
    obj = torch.load(ckpt_path, map_location=device)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
        arch = obj.get("arch", {}) or {}
        depth = int(arch.get("depth", depth_arg if depth_arg is not None else 2))
        n_classes = int(arch.get("n_classes", n_classes_arg if n_classes_arg is not None else 10))
        label_names = obj.get("label_names", None)
        threshold = float(obj.get("sigmoid_threshold", 0.5))
    else:
        if depth_arg is None or n_classes_arg is None:
            raise ValueError("State dict without arch. Please pass --depth and --n_classes.")
        state_dict = obj
        depth, n_classes = depth_arg, n_classes_arg
        label_names, threshold = None, 0.5

    model = ConvNet(depth=depth, n_classes=n_classes).to(device).eval()
    model.load_state_dict(state_dict)
    return model, depth, n_classes, label_names, threshold

def main():
    ap = argparse.ArgumentParser(description="Multilabel ConvNet inference with auto-picked checkpoint.")
    ap.add_argument("--ckpt", type=str, default="Predict/model_best.pt",
                    help="Path to .pt. Default: Predict/model_best.pt; else tries ./checkpoints/")
    ap.add_argument("--weights_dir", type=str, default="checkpoints",
                    help="Where to search for weights if --ckpt not provided")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_npy", type=str, help="Path to a single .npy file")
    g.add_argument("--input_csv", type=str, help="CSV with column 'path' pointing to .npy files")
    ap.add_argument("--output_csv", type=str, default="predictions.csv", help="Output CSV path")
    ap.add_argument("--threshold", type=float, default=None, help="Override sigmoid threshold (default: from checkpoint or 0.5)")
    ap.add_argument("--device", type=str, choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--depth", type=int, default=None, help="Required if the checkpoint stores state_dict only")
    ap.add_argument("--n_classes", type=int, default=None, help="Required if the checkpoint stores state_dict only")
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")

    if args.ckpt and Path(args.ckpt).exists():
        ckpt_path = Path(args.ckpt)
    else:
        best_path = Path(args.weights_dir) / "best.pt"
        ckpt_path = best_path if best_path.exists() else _find_best_ckpt(Path(args.weights_dir))

    model, depth, n_classes, label_names, thr_ckpt = _load_model_and_meta(
        ckpt_path, device, depth_arg=args.depth, n_classes_arg=args.n_classes
    )
    threshold = float(thr_ckpt if args.threshold is None else args.threshold)
    cols, _ = _resolve_label_names(n_classes, label_names)
    
    if args.input_npy:
        arr = np.load(args.input_npy)
        probs, preds = _predict_array(model, device, arr, threshold)
        df = pd.DataFrame([probs], columns=cols)
        for i, c in enumerate(cols):
            df[f"pred_{c}"] = int(preds[i])
        df.insert(0, "path", args.input_npy)
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"[OK] ckpt={ckpt_path}  saved={args.output_csv}")
        return

    df_in = pd.read_csv(args.input_csv)
    if "path" not in df_in.columns:
        raise ValueError("input_csv must contain column 'path'")
    rows = []
    for p in df_in["path"].tolist():
        arr = np.load(p)
        probs, preds = _predict_array(model, device, arr, threshold)
        rows.append((p, probs, preds))

    proba_mat = np.vstack([r[1] for r in rows])
    pred_mat  = np.vstack([r[2] for r in rows]).astype(int)
    
    df_out = pd.DataFrame(proba_mat, columns=cols)
    for i, c in enumerate(cols):
        df_out[f"pred_{c}"] = pred_mat[:, i]
    df_out.insert(0, "path", [r[0] for r in rows])

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"[OK] ckpt={ckpt_path}  saved={args.output_csv}")

if __name__ == "__main__":
    main()





