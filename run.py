#!/usr/bin/env python3
import os
os.environ["MAGNUM_LOG"] = "quiet"
os.environ["HABITAT_SIM_LOG"] = "quiet"

import numpy as np
import argparse
import json
from habitat.datasets import make_dataset
from VLN_CE.vlnce_baselines.config.default import get_config
from my_agent import evaluate_agent

CROSS_FLOOR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "cross_floor_episodes",
)

_FILTERS = {
    "r2r-100": "r2r_v1-2_opennav100_cross_floor.json",
    "r2r-all": "r2r_v1-3_cross_floor.json",
    "rxr-100": "rxr_opennav100_guide_cross_floor.json",
    "rxr-all": "rxr_val_unseen_guide_cross_floor.json",
}

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
        choices=["r2r-100", "r2r-all", "rxr-100", "rxr-all"],
        help="Only run cross-floor episodes",
    )
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, split_num: str, split_id: str, result_path: str,
            cross_floor_filter: str = None, opts=None) -> None:
    config = get_config(exp_config, opts)
    dataset = make_dataset(id_dataset=config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    np.random.seed(42)
    dataset_split = dataset.get_splits(split_num)[split_id]

    if cross_floor_filter is not None:
        filename = _FILTERS[cross_floor_filter]
        filepath = os.path.join(CROSS_FLOOR_DIR, filename)
        with open(filepath) as f:
            cross_ids = set(json.load(f))
        # Also build string versions for matching
        cross_ids_str = set(str(x) for x in cross_ids)
        before = len(dataset_split.episodes)
        dataset_split.episodes = [
            ep for ep in dataset_split.episodes
            if ep.episode_id in cross_ids
            or str(ep.episode_id) in cross_ids_str
            or str(ep.info.get("trajectory_id", "")) in cross_ids_str
        ]
        print(f"Cross-floor filter [{cross_floor_filter}]: {before} -> {len(dataset_split.episodes)} episodes")

    evaluate_agent(config, split_id, dataset_split, result_path)




if __name__ == "__main__":
    main()
