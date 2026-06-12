"""Thin staircase detector interface backed by GroundingDINO."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image


STAIR_LABELS = ("staircase", "stairs", "steps")


class StaircaseDetector:
    """Detect stair-like regions with a minimal backend wrapper."""

    def __init__(self, model_source: Optional[str] = None):
        self.model_source = self._resolve_model_source(model_source)
        self.device = self._resolve_device()
        self.text_threshold = 0.20
        self._processor = None
        self._model = None

    @staticmethod
    def available() -> bool:
        try:
            import torch  # noqa: F401
            from transformers import (  # noqa: F401
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
            )
        except Exception:
            return False
        return True

    def detect(
        self,
        image_rgb: np.ndarray,
        labels: Sequence[str] = STAIR_LABELS,
        threshold: float = 0.50,
        max_det: int = 10,
    ) -> List[Dict[str, Any]]:
        if not self.available():
            return []

        prompt_labels = []
        seen = set()
        for label in labels:
            text = str(label).strip().lower()
            if text and text not in seen:
                seen.add(text)
                prompt_labels.append(text)
        if not prompt_labels:
            return []

        processor, model = self._get_components()
        image = Image.fromarray(np.asarray(image_rgb).astype(np.uint8))
        prompt_text = ". ".join(prompt_labels)
        if not prompt_text.endswith("."):
            prompt_text += "."
        inputs = processor(images=image, text=prompt_text, return_tensors="pt")
        model_inputs = {
            key: (value.to(model.device) if hasattr(value, "to") else value)
            for key, value in inputs.items()
        }

        import torch

        with torch.no_grad():
            outputs = model(**model_inputs)

        post_processor = getattr(
            processor,
            "post_process_grounded_object_detection",
            None,
        )
        if post_processor is None:
            image_processor = getattr(processor, "image_processor", None)
            post_processor = getattr(
                image_processor,
                "post_process_grounded_object_detection",
                None,
            )
        if post_processor is None:
            return []

        kwargs = self._build_post_process_kwargs(
            post_processor,
            outputs=outputs,
            input_ids=model_inputs["input_ids"],
            conf=float(threshold),
            text_threshold=float(self.text_threshold),
            target_sizes=[(int(image.height), int(image.width))],
        )
        results = post_processor(**kwargs)
        result = results[0] if results else {}
        boxes = result.get("boxes") or []
        scores = result.get("scores") or []
        text_labels = result.get("text_labels")
        if text_labels is None:
            text_labels = result.get("labels") or []

        detections: List[Dict[str, Any]] = []
        for box, score, label in list(zip(boxes, scores, text_labels))[: max(1, int(max_det))]:
            if hasattr(box, "tolist"):
                box = box.tolist()
            if hasattr(score, "item"):
                score = score.item()
            detections.append(
                {
                    "label": str(label).strip().lower(),
                    "confidence": float(score),
                    "bbox": [float(v) for v in box],
                }
            )
        return detections

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _build_post_process_kwargs(
        post_processor,
        *,
        outputs,
        input_ids,
        conf: float,
        text_threshold: float,
        target_sizes,
    ) -> Dict[str, Any]:
        params = inspect.signature(post_processor).parameters
        kwargs: Dict[str, Any] = {
            "outputs": outputs,
            "input_ids": input_ids,
            "target_sizes": target_sizes,
        }
        if "box_threshold" in params:
            kwargs["box_threshold"] = float(conf)
        elif "threshold" in params:
            kwargs["threshold"] = float(conf)
        elif "score_threshold" in params:
            kwargs["score_threshold"] = float(conf)
        else:
            raise RuntimeError("Unsupported GroundingDINO post-process signature")
        if "text_threshold" in params:
            kwargs["text_threshold"] = float(text_threshold)
        return kwargs

    def _get_components(self):
        if self._processor is not None and self._model is not None:
            return self._processor, self._model

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            self.model_source,
            local_files_only=True,
            use_fast=False,
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_source,
            local_files_only=True,
        )
        model.to(self.device)
        model.eval()
        self._processor = processor
        self._model = model
        return processor, model

    def _resolve_model_source(self, model_source: Optional[str]) -> str:
        if model_source:
            candidate = Path(model_source).expanduser()
            if candidate.exists():
                return str(candidate.resolve())

        search_roots = [
            Path(__file__).resolve().parents[3] / "vln-zero.github.io" / "vln-lg" / "models" / "grounding-dino-base",
            Path(__file__).resolve().parents[1] / "models" / "grounding-dino-base",
        ]
        for candidate in search_roots:
            if candidate.exists():
                return str(candidate.resolve())
        raise FileNotFoundError(
            "GroundingDINO model directory not found. "
            "Pass --ssa-detector-model-source or place the model under the reference repo."
        )
