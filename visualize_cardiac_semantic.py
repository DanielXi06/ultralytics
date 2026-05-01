from __future__ import annotations

import argparse
import ast
import random
import re
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops

IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
SEMANTIC_PALETTE = {
    0: (0, 0, 0),
    1: (0, 114, 255),
    2: (86, 180, 233),
    3: (0, 200, 83),
    4: (255, 159, 28),
    5: (199, 0, 57),
    6: (156, 39, 176),
    7: (121, 85, 72),
}
BOX_PALETTE = [
    (56, 56, 255),
    (151, 157, 255),
    (31, 112, 255),
    (29, 178, 255),
    (49, 210, 207),
    (10, 249, 72),
]

# -----------------------------------------------------------------------------
# Direct-run configuration
# Edit these values on your Linux server, then run this file directly.
# Example:
#   WEIGHTS_PATH = "/data/xxx/best.pt"
#   DATA_YAML_PATH = "/data/xxx/data.yaml"
#   INFERENCE_SCRIPT_PATH = "/data/xxx/inference_new.py"
#   GPU_ID = 1
# -----------------------------------------------------------------------------
WEIGHTS_PATH = "best.pt"
DATA_YAML_PATH = "data.yaml"
INFERENCE_SCRIPT_PATH = "inference_new.py"

# Set to an integer GPU id such as 0/1/2 to use a specific GPU.
# Set to None to use CPU.
GPU_ID = 0

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
TRAIN_SAMPLE_COUNT = 20
TEST_SAMPLE_COUNT = 20
RANDOM_SEED = 20260416
MASK_ALPHA = 0.45
OUTPUT_DIR = "runs/cardiac_semantic_vis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the auxiliary cardiac semantic branch on sampled train/test images."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(WEIGHTS_PATH),
        help="Path to best.pt checkpoint.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(DATA_YAML_PATH),
        help="Path to dataset yaml.",
    )
    parser.add_argument(
        "--inference-script",
        type=Path,
        default=Path(INFERENCE_SCRIPT_PATH),
        help="Path to inference_new.py that contains inference_source_list.",
    )
    parser.add_argument("--imgsz", type=int, default=IMAGE_SIZE, help="Inference image size.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu" if GPU_ID is None else str(GPU_ID),
        help="Torch device, e.g. 0, cuda:0, cpu. Direct-run default comes from GPU_ID above.",
    )
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Detection IoU threshold.")
    parser.add_argument("--train-count", type=int, default=TRAIN_SAMPLE_COUNT, help="Number of train images to sample.")
    parser.add_argument("--test-count", type=int, default=TEST_SAMPLE_COUNT, help="Number of test images to sample.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducible sampling.")
    parser.add_argument("--alpha", type=float, default=MASK_ALPHA, help="Mask overlay alpha in [0, 1].")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help="Directory for visualization outputs.",
    )
    return parser.parse_args()


def path_is_absolute(path_str: str) -> bool:
    return PureWindowsPath(path_str).is_absolute() or PurePosixPath(path_str).is_absolute()


def resolve_path(path_value: str | Path, root: str | Path | None = None) -> Path:
    path_str = str(path_value)
    if root is not None and not path_is_absolute(path_str):
        return Path(root) / path_str
    return Path(path_str)


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return cleaned or "image"


def load_yaml_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file {path} did not produce a dictionary.")
    return data


def parse_inference_sources(script_path: Path) -> list[str]:
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(text, filename=str(script_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "inference_source_list":
                value = ast.literal_eval(node.value)
                if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                    raise ValueError("inference_source_list must be a literal list[str].")
                return value
    raise ValueError(f"Could not find inference_source_list in {script_path}.")


def collect_images_from_entry(entry: str | Path, root: str | Path | None = None) -> list[Path]:
    path = resolve_path(entry, root)
    if path.is_file():
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return [path]
        if path.suffix.lower() == ".txt":
            images = []
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    images.extend(collect_images_from_entry(line, path.parent))
            return images
        return []
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return []


def collect_images_from_data_entry(entry, root: str | Path | None = None) -> list[Path]:
    if isinstance(entry, str):
        return collect_images_from_entry(entry, root)
    if isinstance(entry, (list, tuple)):
        images = []
        for item in entry:
            images.extend(collect_images_from_data_entry(item, root))
        return images
    return []


def deduplicate_paths(paths: list[Path]) -> list[Path]:
    unique = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def sample_paths(paths: list[Path], count: int, seed: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) <= count:
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(paths, count), key=lambda p: str(p))


def load_grayscale_mask(mask_path: Path, target_shape: tuple[int, int]) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape != target_shape:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.uint8)


def build_semantic_gt_path(data_cfg: dict, image_path: Path, split: str) -> Path | None:
    data_root = data_cfg.get("path")
    split_key = f"sem_{split}"
    if split_key in data_cfg:
        sem_root = resolve_path(data_cfg[split_key], data_root)
        return sem_root / f"{image_path.stem}.png"
    if "sem_masks" in data_cfg:
        sem_root = resolve_path(data_cfg["sem_masks"], data_root) / split
        return sem_root / f"{image_path.stem}.png"
    return None


def select_device(device_arg: str) -> torch.device:
    if not device_arg:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device_arg.isdigit():
        return torch.device(f"cuda:{device_arg}")
    return torch.device(device_arg)


def normalize_stride(stride_value) -> int:
    if isinstance(stride_value, torch.Tensor):
        if stride_value.ndim == 0:
            return int(stride_value.item())
        return int(stride_value.max().item())
    if isinstance(stride_value, (list, tuple)):
        return int(max(stride_value))
    return int(stride_value)


class CardiacSemanticRunner:
    def __init__(self, weights: Path, imgsz: int, device: str, conf: float, iou: float):
        self.yolo = YOLO(str(weights))
        self.model = self.yolo.model
        self.device = select_device(device)
        self.model.to(self.device).eval()
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = getattr(self.model, "names", getattr(self.yolo, "names", {}))
        self.head = self.model.model[-1]
        self.stride = normalize_stride(getattr(self.model, "stride", torch.tensor([32])))
        if not hasattr(self.head, "sem_proto"):
            raise TypeError("The loaded model head does not expose a semantic branch through sem_proto.")
        if getattr(self.head.sem_proto, "semseg", None) is None:
            raise RuntimeError("The semantic branch appears to be fused away. Please load an unfused .pt checkpoint.")
        self._head_inputs: list[torch.Tensor] | None = None
        self._hook = self.head.register_forward_hook(self._capture_head_inputs)

    def close(self) -> None:
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def _capture_head_inputs(self, module, inputs, outputs) -> None:
        if not inputs:
            self._head_inputs = None
            return
        features = inputs[0]
        if isinstance(features, (list, tuple)):
            self._head_inputs = [feat.detach() for feat in features]
        else:
            self._head_inputs = None

    def preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        transformed = LetterBox(self.imgsz, auto=False, stride=self.stride)(image=image_bgr)
        tensor = transformed[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor)
        tensor = torch.from_numpy(tensor).to(self.device).float() / 255.0
        return tensor.unsqueeze(0)

    @torch.inference_mode()
    def infer(self, image_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        self._head_inputs = None
        image_tensor = self.preprocess(image_bgr)
        raw_preds = self.model(image_tensor)
        if self._head_inputs is None:
            raise RuntimeError("Failed to capture semantic branch inputs from the detection head.")

        detections = nms.non_max_suppression(
            raw_preds,
            conf_thres=self.conf,
            iou_thres=self.iou,
            max_det=300,
            nc=0,
            end2end=getattr(self.model, "end2end", False),
        )[0]
        if len(detections):
            detections[:, :4] = ops.scale_boxes(image_tensor.shape[2:], detections[:, :4], image_bgr.shape)

        sem_logits = self._forward_semantic_logits(self._head_inputs)
        sem_logits = ops.scale_masks(sem_logits, image_bgr.shape[:2])
        sem_map = sem_logits.argmax(dim=1)[0].byte().cpu().numpy()
        return detections.detach().cpu(), sem_map

    def _forward_semantic_logits(self, features: list[torch.Tensor]) -> torch.Tensor:
        sem_proto = self.head.sem_proto
        feat = features[0]
        for i, refine_layer in enumerate(sem_proto.feat_refine):
            refined = refine_layer(features[i + 1])
            refined = F.interpolate(refined, size=feat.shape[2:], mode="nearest")
            feat = feat + refined
        return sem_proto.semseg(feat)


def draw_detections(image: np.ndarray, detections: torch.Tensor, names: dict) -> np.ndarray:
    canvas = image.copy()
    if detections is None or len(detections) == 0:
        return canvas
    det_array = detections.numpy()
    for det in det_array:
        x1, y1, x2, y2, conf, cls = det[:6]
        cls = int(cls)
        color = BOX_PALETTE[cls % len(BOX_PALETTE)]
        p1 = int(round(x1)), int(round(y1))
        p2 = int(round(x2)), int(round(y2))
        cv2.rectangle(canvas, p1, p2, color, 2, lineType=cv2.LINE_AA)
        label = f"{names.get(cls, cls)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = max(p1[1] - 8, th + 6)
        cv2.rectangle(canvas, (p1[0], y_text - th - 6), (p1[0] + tw + 6, y_text), color, -1)
        cv2.putText(
            canvas,
            label,
            (p1[0] + 3, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return canvas


def colorize_semantic_mask(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls_id in np.unique(mask):
        color = SEMANTIC_PALETTE.get(int(cls_id), SEMANTIC_PALETTE[max(SEMANTIC_PALETTE)])
        canvas[mask == cls_id] = color
    return canvas


def overlay_semantic_mask(image: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    color_mask = colorize_semantic_mask(mask)
    blended = image.copy()
    fg = mask > 0
    if np.any(fg):
        blended[fg] = ((1.0 - alpha) * blended[fg] + alpha * color_mask[fg]).astype(np.uint8)
    return blended


def add_legend(image: np.ndarray, sem_names: dict[int, str], mask: np.ndarray) -> np.ndarray:
    present = [int(x) for x in np.unique(mask) if int(x) != 0]
    if not present:
        return image
    canvas = image.copy()
    box_h = 26 * len(present) + 14
    box_w = 240
    cv2.rectangle(canvas, (10, 10), (10 + box_w, 10 + box_h), (255, 255, 255), -1)
    cv2.rectangle(canvas, (10, 10), (10 + box_w, 10 + box_h), (64, 64, 64), 1)
    y = 34
    for cls_id in present:
        color = SEMANTIC_PALETTE.get(cls_id, SEMANTIC_PALETTE[max(SEMANTIC_PALETTE)])
        name = sem_names.get(cls_id, f"class_{cls_id}")
        cv2.rectangle(canvas, (22, y - 12), (40, y + 6), color, -1)
        pixels = int((mask == cls_id).sum())
        text = f"{cls_id}: {name} ({pixels}px)"
        cv2.putText(canvas, text, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        y += 26
    return canvas


def make_panel(image: np.ndarray, title: str) -> np.ndarray:
    header_h = 36
    panel = np.full((image.shape[0] + header_h, image.shape[1], 3), 255, dtype=np.uint8)
    panel[header_h:] = image
    cv2.putText(panel, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(panel, (0, header_h - 1), (panel.shape[1], header_h - 1), (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def save_visualization(
    image_path: Path,
    output_path: Path,
    image: np.ndarray,
    detections: torch.Tensor,
    pred_mask: np.ndarray,
    sem_names: dict[int, str],
    alpha: float,
    gt_mask: np.ndarray | None = None,
    names: dict | None = None,
) -> None:
    boxes_img = draw_detections(image, detections, names or {})
    overlay = add_legend(overlay_semantic_mask(boxes_img, pred_mask, alpha), sem_names, pred_mask)
    pred_mask_color = add_legend(colorize_semantic_mask(pred_mask), sem_names, pred_mask)
    panels = [
        make_panel(boxes_img, f"Image + Detections: {image_path.name}"),
        make_panel(overlay, "Predicted Overlay"),
        make_panel(pred_mask_color, "Predicted Semantic Mask"),
    ]
    if gt_mask is not None:
        gt_overlay = add_legend(overlay_semantic_mask(image, gt_mask, alpha), sem_names, gt_mask)
        panels.append(make_panel(gt_overlay, "Ground Truth Mask"))
    combined = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def write_manifest(manifest_path: Path, image_paths: list[Path]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(f"{path}\n")


def process_split(
    split_name: str,
    image_paths: list[Path],
    runner: CardiacSemanticRunner,
    output_dir: Path,
    sem_names: dict[int, str],
    alpha: float,
    data_cfg: dict | None = None,
    gt_split_name: str | None = None,
) -> tuple[int, int]:
    saved = 0
    failed = 0
    for idx, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Failed to read image: {image_path}")
            failed += 1
            continue

        try:
            gt_mask = None
            if data_cfg is not None and gt_split_name is not None:
                gt_path = build_semantic_gt_path(data_cfg, image_path, gt_split_name)
                if gt_path is not None:
                    gt_mask = load_grayscale_mask(gt_path, image.shape[:2])

            detections, pred_mask = runner.infer(image)
            out_name = f"{idx:02d}_{sanitize_name(image_path.stem)}.jpg"
            out_path = output_dir / split_name / out_name
            save_visualization(
                image_path=image_path,
                output_path=out_path,
                image=image,
                detections=detections,
                pred_mask=pred_mask,
                sem_names=sem_names,
                alpha=alpha,
                gt_mask=gt_mask,
                names=runner.names,
            )
            saved += 1
            print(f"[{split_name}] {idx}/{len(image_paths)} saved -> {out_path}")
        except Exception as exc:
            failed += 1
            print(f"[WARN] {split_name} failed for {image_path}: {exc}")
    return saved, failed


def main() -> None:
    args = parse_args()
    if not args.weights.exists():
        raise FileNotFoundError(
            f"Weights file not found: {args.weights}. "
            "On the Linux server, pass an explicit path like --weights /path/to/best.pt"
        )
    if not args.data.exists():
        raise FileNotFoundError(
            f"Data yaml not found: {args.data}. "
            "On the Linux server, pass an explicit path like --data /path/to/data.yaml"
        )
    if not args.inference_script.exists():
        raise FileNotFoundError(
            f"Inference script not found: {args.inference_script}. "
            "On the Linux server, pass an explicit path like --inference-script /path/to/inference_new.py"
        )

    data_cfg = load_yaml_file(args.data)
    inference_sources = parse_inference_sources(args.inference_script)

    sem_names_cfg = data_cfg.get("sem_names", {})
    sem_names = {int(k): str(v) for k, v in sem_names_cfg.items()} if isinstance(sem_names_cfg, dict) else {}
    if not sem_names:
        sem_names = {0: "background", 1: "heart_region"}

    data_root = data_cfg.get("path")
    train_entry = data_cfg.get("train")
    if train_entry is None:
        raise ValueError(f"'train' was not found in {args.data}.")

    train_images = deduplicate_paths(collect_images_from_data_entry(train_entry, data_root))
    test_images = deduplicate_paths(
        [img for source in inference_sources for img in collect_images_from_entry(source, args.inference_script.parent)]
    )

    if not train_images:
        raise FileNotFoundError("No train images were found from data.yaml.")
    if not test_images:
        raise FileNotFoundError("No test images were found from inference_new.py.")

    sampled_train = sample_paths(train_images, args.train_count, args.seed)
    sampled_test = sample_paths(test_images, args.test_count, args.seed + 1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(run_dir / "train_manifest.txt", sampled_train)
    write_manifest(run_dir / "test_manifest.txt", sampled_test)

    runner = CardiacSemanticRunner(
        weights=args.weights,
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
    )
    try:
        train_saved, train_failed = process_split(
            split_name="train",
            image_paths=sampled_train,
            runner=runner,
            output_dir=run_dir,
            sem_names=sem_names,
            alpha=args.alpha,
            data_cfg=data_cfg,
            gt_split_name="train",
        )
        test_saved, test_failed = process_split(
            split_name="test",
            image_paths=sampled_test,
            runner=runner,
            output_dir=run_dir,
            sem_names=sem_names,
            alpha=args.alpha,
        )
    finally:
        runner.close()

    print("")
    print(f"Output directory: {run_dir.resolve()}")
    print(f"Train images requested/saved/failed: {args.train_count}/{train_saved}/{train_failed}")
    print(f"Test images requested/saved/failed: {args.test_count}/{test_saved}/{test_failed}")
    print(f"Train manifest: {(run_dir / 'train_manifest.txt').resolve()}")
    print(f"Test manifest: {(run_dir / 'test_manifest.txt').resolve()}")


if __name__ == "__main__":
    main()
