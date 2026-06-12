"""Runtime adapter for the trained SSA pose model."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class SSAPoseRuntime:
    """Lazy SSA inference wrapper.

    The wrapped model predicts a relative robot-frame target pose:
    +x forward, +y left.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        workspace_root: Path,
        module_root: str = "SemanticSpatialAlignmentModule",
        device: str = "auto",
    ):
        self.workspace_root = Path(workspace_root)
        self.checkpoint_path = self._resolve_path(checkpoint_path)
        self.module_root = self._resolve_path(module_root)
        self.device_name = str(device or "auto")
        self._loaded = False
        self._load_error = ""
        self._torch = None
        self._cv2 = None
        self._depth_to_color = None
        self._config = None
        self._model = None
        self._device = None

    def estimate(
        self,
        *,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = {
            "enabled": True,
            "usable": False,
            "label": str(detection.get("label", "")).strip().lower(),
            "confidence": float(detection.get("confidence", 0.0) or 0.0),
            "bbox": list(detection.get("bbox", []) or []),
            "error": "",
        }
        if depth is None:
            result["error"] = "missing_depth"
            return result
        if not self._ensure_loaded():
            result["error"] = self._load_error or "ssa_model_load_failed"
            return result

        try:
            tensors = self._preprocess(rgb=rgb, depth=depth, bbox=result["bbox"])
            torch = self._torch
            with torch.no_grad():
                outputs = self._model(**tensors)
                xy, bearing, yaw, uncertainty = self._decode_pose(outputs)
            x_forward = float(xy[0, 0].detach().cpu().item())
            y_left = float(xy[0, 1].detach().cpu().item())
            yaw_rad = float(yaw[0].detach().cpu().item())
            unc = float(uncertainty[0].detach().cpu().item())
            distance = float(math.hypot(x_forward, y_left))
            bearing_rad = float(math.atan2(y_left, max(1e-6, x_forward)))
            result.update(
                {
                    "usable": bool(math.isfinite(distance)),
                    "x_forward_m": x_forward,
                    "y_left_m": y_left,
                    "right_m": -y_left,
                    "distance_m": distance,
                    "bearing_rad": bearing_rad,
                    "bearing_deg": math.degrees(bearing_rad),
                    "yaw_rad": yaw_rad,
                    "yaw_deg": math.degrees(yaw_rad),
                    "uncertainty": unc,
                }
            )
            return result
        except Exception as exc:
            result["error"] = f"{exc.__class__.__name__}: {exc}"[:240]
            return result

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_error:
            return False
        try:
            if str(self.module_root) not in sys.path:
                sys.path.insert(0, str(self.module_root))
            import cv2
            import torch
            from baseline.config import TrainConfig
            from baseline.depth_utils import depth_to_color
            from baseline.model import DualResNetPoseRegressor

            if self.device_name == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(self.device_name)

            checkpoint = torch.load(
                str(self.checkpoint_path),
                map_location=device,
                weights_only=False,
            )
            config = checkpoint.get("config") or TrainConfig()
            model = DualResNetPoseRegressor(
                backbone=getattr(config, "backbone", "resnet18"),
                pretrained=False,
                dropout=float(getattr(config, "dropout", 0.2)),
                yaw_bins=int(getattr(config, "yaw_bins", 12)),
                distance_bins=int(getattr(config, "distance_bins", 6)),
                bearing_bins=int(getattr(config, "bearing_bins", 12)),
            ).to(device)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state)
            model.eval()

            self._torch = torch
            self._cv2 = cv2
            self._depth_to_color = depth_to_color
            self._config = config
            self._model = model
            self._device = device
            self._loaded = True
            return True
        except Exception as exc:
            self._load_error = f"{exc.__class__.__name__}: {exc}"[:240]
            return False

    def _preprocess(self, *, rgb: np.ndarray, depth: np.ndarray, bbox) -> Dict[str, Any]:
        cv2 = self._cv2
        torch = self._torch
        cfg = self._config
        image_h = int(getattr(cfg, "image_height", 480))
        image_w = int(getattr(cfg, "image_width", 640))

        rgb_arr = np.asarray(rgb)
        if rgb_arr.ndim == 2:
            rgb_arr = np.repeat(rgb_arr[:, :, None], 3, axis=2)
        rgb_arr = rgb_arr[:, :, :3].astype(np.uint8)
        depth_arr = np.asarray(depth, dtype=np.float32)
        if depth_arr.ndim == 3:
            depth_arr = depth_arr[:, :, 0]

        src_h, src_w = rgb_arr.shape[:2]
        rgb_resized = cv2.resize(rgb_arr, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth_arr, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
        depth_color = self._depth_to_color(
            depth_resized,
            max_depth=float(getattr(cfg, "max_depth", 10.0)),
        )

        x1, y1, x2, y2 = self._scaled_bbox(bbox, src_w, src_h, image_w, image_h)
        crop_rgb = rgb_resized[y1:y2, x1:x2]
        crop_depth = depth_color[y1:y2, x1:x2]
        if crop_rgb.size == 0 or crop_depth.size == 0:
            crop_rgb = rgb_resized
            crop_depth = depth_color
        crop_rgb = cv2.resize(crop_rgb, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        crop_depth = cv2.resize(crop_depth, (image_w, image_h), interpolation=cv2.INTER_LINEAR)

        return {
            "rgb": self._to_tensor(rgb_resized).to(self._device),
            "depth_color": self._to_tensor(depth_color).to(self._device),
            "crop_rgb": self._to_tensor(crop_rgb).to(self._device),
            "crop_depth_color": self._to_tensor(crop_depth).to(self._device),
        }

    def _to_tensor(self, image: np.ndarray):
        torch = self._torch
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def _decode_pose(self, outputs: Dict[str, Any]):
        torch = self._torch
        cfg = self._config
        device = outputs["distance_residual"].device
        centers = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5],
            dtype=torch.float32,
            device=device,
        )
        distance_bin = outputs["distance_bin_logits"].argmax(dim=1)
        distance = (centers[distance_bin] + outputs["distance_residual"]).clamp_min(0.0)
        bearing = self._decode_angle_bin(
            outputs["bearing_bin_logits"],
            outputs["bearing_residual"],
            int(getattr(cfg, "bearing_bins", 12)),
        )
        yaw = self._decode_angle_bin(
            outputs["yaw_bin_logits"],
            outputs["yaw_residual"],
            int(getattr(cfg, "yaw_bins", 12)),
        )
        xy = torch.stack([distance * torch.cos(bearing), distance * torch.sin(bearing)], dim=1)
        uncertainty = torch.exp(0.5 * outputs["log_var"].clamp(-6.0, 4.0))
        return xy, bearing, yaw, uncertainty

    def _decode_angle_bin(self, logits, residual, bins: int):
        torch = self._torch
        bin_width = 2.0 * math.pi / int(bins)
        pred_bin = logits.argmax(dim=1).float()
        center = -math.pi + (pred_bin + 0.5) * bin_width
        angle = center + residual
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    @staticmethod
    def _scaled_bbox(bbox, src_w: int, src_h: int, dst_w: int, dst_h: int):
        if not bbox or len(bbox) != 4:
            return 0, 0, dst_w, dst_h
        sx = dst_w / max(1.0, float(src_w))
        sy = dst_h / max(1.0, float(src_h))
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        pad_x = 0.10 * max(1.0, x2 - x1)
        pad_y = 0.10 * max(1.0, y2 - y1)
        x1 = int(max(0, math.floor(x1 - pad_x)))
        y1 = int(max(0, math.floor(y1 - pad_y)))
        x2 = int(min(dst_w, math.ceil(x2 + pad_x)))
        y2 = int(min(dst_h, math.ceil(y2 + pad_y)))
        if x2 <= x1 or y2 <= y1:
            return 0, 0, dst_w, dst_h
        return x1, y1, x2, y2

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.workspace_root / candidate).resolve()
