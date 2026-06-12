"""Low-intrusion SSA proposal and takeover controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .detector import StaircaseDetector
from .planner import SSAPlan, SimulatorAStarPlanner
from .runtime import SSAPoseRuntime


STAIR_KEYWORDS = (
    "stair",
    "stairs",
    "step",
    "steps",
    "staircase",
    "upstairs",
    "downstairs",
)


@dataclass
class SSAStatus:
    available: bool
    used: bool
    active: bool
    error: str = ""


class SSAController:
    """Owns SSA proposal gating and the single-use takeover lifecycle."""

    def __init__(
        self,
        *,
        enabled: bool,
        workspace_root: Path,
        checkpoint_path: str = "",
        detect_threshold: float = 0.50,
        uncertainty_threshold: float = 0.50,
        max_distance_m: float = 3.0,
        detector_model_source: Optional[str] = None,
    ):
        self.enabled = bool(enabled and checkpoint_path)
        self.detect_threshold = float(detect_threshold)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.max_distance_m = float(max_distance_m)
        self.workspace_root = Path(workspace_root)
        self.detector = None
        self.runtime = None
        if self.enabled:
            self.detector = StaircaseDetector(detector_model_source)
            self.runtime = SSAPoseRuntime(
                checkpoint_path,
                workspace_root=self.workspace_root,
            )
        self.planner = SimulatorAStarPlanner()
        self.used_this_episode = False
        self.takeover_active = False
        self.pending_actions = []
        self.plan: Optional[SSAPlan] = None
        self.last_proposal: Dict[str, Any] = {}
        self.last_error = ""

    def reset(self) -> None:
        self.used_this_episode = False
        self.takeover_active = False
        self.pending_actions = []
        self.plan = None
        self.last_proposal = {}
        self.last_error = ""

    def current_status(self) -> SSAStatus:
        return SSAStatus(
            available=bool(self.last_proposal.get("available", False)),
            used=bool(self.used_this_episode),
            active=bool(self.takeover_active),
            error=str(self.last_error),
        )

    def should_offer(self, instruction: str, previous_output: str = "", previous_plan: str = "") -> bool:
        text = " ".join(
            [
                str(instruction or "").lower(),
                str(previous_output or "").lower(),
                str(previous_plan or "").lower(),
            ]
        )
        return any(token in text for token in STAIR_KEYWORDS)

    def update_proposal(
        self,
        *,
        instruction: str,
        previous_output: str,
        previous_plan: str,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        proposal = {
            "available": False,
            "used": bool(self.used_this_episode),
            "active": bool(self.takeover_active),
            "reason": "",
        }
        self.last_error = ""
        if not self.enabled:
            proposal["reason"] = "disabled"
        elif self.used_this_episode:
            proposal["reason"] = "already_used"
        elif not self.should_offer(instruction, previous_output, previous_plan):
            proposal["reason"] = "not_stair_context"
        elif depth is None:
            proposal["reason"] = "missing_depth"
        else:
            detections = self.detector.detect(
                rgb,
                threshold=self.detect_threshold,
                max_det=10,
            )
            detections = [
                det for det in detections
                if str(det.get("label", "")).strip().lower() in {"staircase", "stairs", "steps"}
                and float(det.get("confidence", 0.0) or 0.0) >= self.detect_threshold
            ]
            if not detections:
                proposal["reason"] = "no_stair_detection"
            else:
                best = max(detections, key=lambda item: float(item.get("confidence", 0.0) or 0.0))
                estimate = self.runtime.estimate(rgb=rgb, depth=depth, detection=best)
                proposal.update({"estimate": estimate})
                gating_error = self._proposal_rejection_reason(estimate)
                if gating_error:
                    proposal["reason"] = gating_error
                else:
                    proposal["available"] = True
                    proposal["reason"] = "ok"
        self.last_proposal = proposal
        self.last_error = str(proposal.get("reason", ""))
        return proposal

    def maybe_start_takeover(self, *, delegate: bool, env) -> bool:
        if not delegate or not self.last_proposal.get("available", False):
            return False
        estimate = dict(self.last_proposal.get("estimate", {}) or {})
        self.used_this_episode = True
        self.plan = self.planner.build_plan(env, estimate)
        if self.plan.error or not self.plan.actions:
            self.takeover_active = False
            self.pending_actions = []
            self.last_error = self.plan.error or "ssa_plan_empty"
            return False
        self.pending_actions = list(self.plan.actions)
        self.takeover_active = True
        self.last_error = ""
        return True

    def next_takeover_action(self, env) -> Optional[int]:
        if not self.takeover_active or self.plan is None:
            return None
        if self.planner.reached_target(env, self.plan.target_position, self.plan.target_yaw_deg):
            self._finish_takeover()
            return None
        if not self.pending_actions:
            self.last_error = "ssa_takeover_exhausted_without_reaching_target"
            self._finish_takeover()
            return None
        return int(self.pending_actions.pop(0))

    def notify_action_result(self, collision: bool) -> None:
        if self.takeover_active and collision:
            self.last_error = "ssa_collision_failure"
            self._finish_takeover()

    def prompt_status_text(self) -> str:
        status = self.current_status()
        if not self.enabled:
            return "SSA stair takeover: unavailable."
        return (
            "SSA stair takeover: "
            f"{'available' if status.available else 'unavailable'}. "
            "If available, you may delegate the current stair-related local phase to SSA. "
            "If you delegate, SSA will autonomously handle local stair alignment/approach until success or failure."
        )

    def _finish_takeover(self) -> None:
        self.takeover_active = False
        self.pending_actions = []

    def _proposal_rejection_reason(self, estimate: Dict[str, Any]) -> str:
        if not bool(estimate.get("usable", False)):
            return str(estimate.get("error", "") or "ssa_unusable")
        if float(estimate.get("confidence", 0.0) or 0.0) < self.detect_threshold:
            return "low_detection_confidence"
        if float(estimate.get("uncertainty", 1e9) or 1e9) > self.uncertainty_threshold:
            return "high_uncertainty"
        if float(estimate.get("x_forward_m", 0.0) or 0.0) <= 0.0:
            return "target_behind_agent"
        if float(estimate.get("distance_m", 1e9) or 1e9) > self.max_distance_m:
            return "target_too_far"
        return ""
