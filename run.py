#!/usr/bin/env python3
import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import numpy as np
import argparse
import sys
from pathlib import Path
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from my_agent import evaluate_agent

CWD = Path(__file__).resolve().parent
PROJECT_ROOT = CWD.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from shared.cross_floor_filter import get_cross_floor_episode_ids
from shared.episode_filter import parse_episode_ids
from shared.evaluation_selection import filter_episode_objects
from shared.resume_utils import collect_completed_episode_ids


def enable_depth_sensor_for_ssa(config):
    """SSA needs RGB-D while the zero-shot configs expose RGB only."""
    config.defrost()
    simulator = config.TASK_CONFIG.SIMULATOR
    sensors = list(simulator.AGENT_0.SENSORS)
    if "DEPTH_SENSOR" not in sensors:
        sensors.append("DEPTH_SENSOR")
        simulator.AGENT_0.SENSORS = sensors
    depth_sensor = simulator.DEPTH_SENSOR
    rgb_sensor = simulator.RGB_SENSOR
    depth_sensor.WIDTH = int(getattr(rgb_sensor, "WIDTH", 640))
    depth_sensor.HEIGHT = int(getattr(rgb_sensor, "HEIGHT", 480))
    depth_sensor.HFOV = int(getattr(rgb_sensor, "HFOV", 90))
    depth_sensor.TYPE = "HabitatSimDepthSensor"
    if hasattr(rgb_sensor, "POSITION"):
        depth_sensor.POSITION = list(rgb_sensor.POSITION)
    config.freeze()
    return config

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "--split-num",
        type=int,
        required=True,
        help="chunks of evaluation"
    )
    
    parser.add_argument(
        "--split-id",
        type=int,
        required=True,
        help="chunks ID of evaluation"

    )
    parser.add_argument(
        "--result-path",
        type=str,
        required=True,
        help="location to save results"
    )
    parser.add_argument(
        "--cross-floor-filter",
        type=str,
        default=None,
        choices=["r2r-100", "r2r-100-0.5", "r2r-all", "rxr-100", "rxr-all"],
        help="Only run cross-floor episodes",
    )
    parser.add_argument(
        "--ssa-guidance",
        action="store_true",
        help="Enable single-use SSA stair takeover.",
    )
    parser.add_argument(
        "--ssa-checkpoint",
        type=str,
        default="",
        help="Path to the trained SSA checkpoint.",
    )
    parser.add_argument(
        "--ssa-detect-threshold",
        type=float,
        default=0.30,
        help="Minimum stair detection confidence before SSA proposal.",
    )
    parser.add_argument(
        "--ssa-detector-model-source",
        type=str,
        default="",
        help="Optional local GroundingDINO model directory.",
    )
    parser.add_argument(
        "--filter-behind",
        action="store_true",
        help="Reject SSA proposals where the predicted target is behind the agent (x_forward_m < 0).",
    )
    parser.add_argument(
        "--oracle-exit-enable",
        action="store_true",
        help="Use expert-path oracle exit as SSA diagnostic fallback.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes that already have per-episode result logs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override ENVIRONMENT.MAX_EPISODE_STEPS (default: from config yaml).",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="Comma-separated episode ids to evaluate, e.g. 1413,1370,1371.",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, result_path: str,
            cross_floor_filter: str = None, ssa_guidance: bool = False,
            ssa_checkpoint: str = "", ssa_detect_threshold: float = 0.30,
            ssa_detector_model_source: str = "", filter_behind: bool = False,
            max_steps: int = None,
            resume: bool = False, episode_id: str = None,
            oracle_exit_enable: bool = False,
            opts=None) -> None:
    config = get_config(exp_config, opts)
    if max_steps is not None:
        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS = int(max_steps)
        config.TASK_CONFIG.TASK.TOP_DOWN_MAP_VLNCE.MAX_EPISODE_STEPS = int(max_steps)
        config.freeze()
        print(f"[CONFIG] MAX_EPISODE_STEPS overridden to {max_steps}")
    if ssa_guidance:
        if not str(ssa_checkpoint or "").strip():
            raise ValueError("--ssa-guidance requires an explicit --ssa-checkpoint")
        print(f"[SSA] enabled | checkpoint={ssa_checkpoint} | detect_threshold={ssa_detect_threshold}")
        config = enable_depth_sensor_for_ssa(config)
    else:
        print("[SSA] disabled")
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num)[split_id]

    if cross_floor_filter is not None:
        before = len(dataset_split.episodes)
        dataset_split.episodes = filter_episode_objects(
            dataset_split.episodes,
            cross_floor_ids=get_cross_floor_episode_ids(cross_floor_filter),
        )
        print(f"Cross-floor filter [{cross_floor_filter}]: {before} -> {len(dataset_split.episodes)} episodes")

    requested_episode_ids = parse_episode_ids(episode_id)
    if requested_episode_ids:
        before = len(dataset_split.episodes)
        dataset_split.episodes = filter_episode_objects(
            dataset_split.episodes,
            requested_ids=requested_episode_ids,
        )
        print(
            f"Episode-id filter [{','.join(requested_episode_ids)}]: "
            f"{before} -> {len(dataset_split.episodes)} episodes"
        )

    if resume:
        completed_ids = collect_completed_episode_ids(result_path)
        before = len(dataset_split.episodes)
        dataset_split.episodes = filter_episode_objects(
            dataset_split.episodes,
            completed_ids=completed_ids,
        )
        print(
            f"Resume filter: {before} -> {len(dataset_split.episodes)} episodes "
            f"(skipped {before - len(dataset_split.episodes)} completed from {Path(result_path) / 'log'})"
        )

    evaluate_agent(
        config,
        split_id,
        dataset_split,
        result_path,
        ssa_guidance=ssa_guidance,
        ssa_checkpoint=ssa_checkpoint,
        ssa_detect_threshold=ssa_detect_threshold,
        ssa_detector_model_source=ssa_detector_model_source,
        filter_behind=filter_behind,
        oracle_exit_enable=oracle_exit_enable,
    )




if __name__ == "__main__":
    main()
