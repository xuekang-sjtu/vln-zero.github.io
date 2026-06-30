# IMPORTANT: CHECK LINES 134, 142, AND 944
# FOR SETTINGS YOU MAY WANT TO CONFIGURE
# BEFORE RUNNING EVALUATION.

import json
import numpy as np
from habitat import Env
from habitat.core.agent import Agent
from tqdm import trange
import os
import re
import cv2
import imageio
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
import random
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
from openai import OpenAI
import pickle
from pathlib import Path
import networkx as nx
import math
import time
import statistics
import sys
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.eval_metrics import format_episode_metric
from shared.ssa import SSAController
from shared.ssa.oracle import select_oracle_exit_for_episode
from shared.ssa.trajectory import save_trajectory_debug

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def openai_api_calculate_cost(usage):
    model_pricing = {
        'prompt': 0.002,
        'cached': 0.0005,
        'completion': 0.008,
    }

    prompt_cost = usage.prompt_tokens * model_pricing['prompt'] / 1000
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens = getattr(prompt_details, "cached_tokens", 0) if prompt_details is not None else 0
    cached_cost = cached_tokens * model_pricing['cached'] / 1000
    completion_cost = usage.completion_tokens * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost + cached_cost
    #print(f"\nTokens used:  {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} completion = {usage['total_tokens']:,} tokens")
    #print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return total_cost

def find_nearest_waypoint_to(curr_pos, curr_sg, thresh=None):
    
    min_dist = 10000000.
    min_dist_node = None

    for node in curr_sg.nodes(data=True):
        squared_diff_sum = 0
        for i in [0, 2]: # ignore y direction, single floor example
            squared_diff_sum += (curr_pos[i] - node[1]['position'][i]) ** 2
        
        dist = math.sqrt(squared_diff_sum)
        if (dist < min_dist):
            min_dist = dist
            min_dist_node = node[0]
    
    if ((thresh) and (min_dist > thresh)):
        return None

    return min_dist_node

def check_if_pos_near_node(curr_pos, node_to_check, thresh):
    squared_diff_sum = 0
    for i in [0, 2]: # ignore y direction, single floor example
        squared_diff_sum += (curr_pos[i] - node_to_check[i]) ** 2
        
        dist = math.sqrt(squared_diff_sum)
    
    if ((dist > thresh)):
        return False

    return True

def try_get_cached_path(node1, node2, curr_sg):
    eligible_edges = [(u, v) for u, v, d in curr_sg.edges(data=True) if 'cache_info' in d]
    G_eligible = curr_sg.edge_subgraph(eligible_edges)
    try:
        path = nx.shortest_path(G_eligible, source=node1, target=node2)
        #print(f"Shortest path found: {path}")
    except:
        path = None
        #print("No path found between the specified nodes using eligible edges.")

    return path

def find_best_cached_jump(graph, start_node, goal_node):
    """
    Finds the first node to navigate to via a beneficial cached path.
    Returns the node if one is found, otherwise returns None.
    """
    try:
        # Pre-calculate shortest path lengths from all nodes to the goal as the heuristic h(n).
        h = nx.shortest_path_length(graph, target=goal_node)
    except nx.NetworkXNoPath:
        # Handle cases where the goal is unreachable
        return None

    best_next_node = None
    best_h_value = float('inf')

    # Get the heuristic value for the start node
    start_h_value = h.get(start_node, float('inf'))

    # Iterate over all neighbors of the start node
    for neighbor in graph.neighbors(start_node):
        if (neighbor == start_node):
            continue
        edge_data = graph.get_edge_data(start_node, neighbor)
        
        # Check if a cached path exists for this edge
        if 'cache_path' in edge_data:
            # Check if this cached path brings us closer to the goal
            neighbor_h_value = h.get(neighbor, float('inf'))
            
            if neighbor_h_value < start_h_value:
                # This is a beneficial cached path. Check if it's the best one so far.
                if neighbor_h_value < best_h_value:
                    best_h_value = neighbor_h_value
                    best_next_node = neighbor
                    
    return best_next_node

def evaluate_agent(
    config,
    split_id,
    dataset,
    result_path,
    ssa_guidance=False,
    ssa_checkpoint="",
    ssa_detect_threshold=0.30,
    ssa_detector_model_source="",
    filter_behind=False,
    oracle_exit_enable=False,
) -> None:
    enable_use_of_cache = False
    enable_adding_to_cache = False # should be false if we do not have enable_use_of_cache

    env = Env(config.TASK_CONFIG, dataset)
    agent = MyGPTAgent(
        result_path,
        workspace_root=WORKSPACE_ROOT,
        ssa_guidance=ssa_guidance,
        ssa_checkpoint=ssa_checkpoint,
        ssa_detect_threshold=ssa_detect_threshold,
        ssa_detector_model_source=ssa_detector_model_source,
        filter_behind=filter_behind,
        oracle_exit_enable=oracle_exit_enable,
    )
    num_episodes = len(env.episodes) # You can customize this to a low number (e.g. 5) to run on a small subset of examples.
    EARLY_STOP_ROTATION = int(getattr(config.EVAL, "EARLY_STOP_ROTATION", 0) or 0)
    EARLY_STOP_STEPS = int(getattr(config.EVAL, "EARLY_STOP_STEPS", 100) or 0)

    target_key = {"distance_to_goal", "success", "spl", "path_length", "oracle_success"}

    count = 0

    # It is HIGHLY RECOMMENDED to create a backup or custom file for your scene graphs
    # and change the path here to that instead.
    SCENE_GRAPHS_PATH = "../datasets/connectivity/connectivity_graphs.pkl"

    for _ in trange(
        num_episodes, desc=getattr(config.EVAL, "IDENTIFICATION", "test") + "-{}".format(split_id)
    ):
        obs = env.reset()
        iter_step = 0
        agent.reset()

        scene_dir_parts = str(config.TASK_CONFIG.SIMULATOR.SCENE).split('/')
        scene_id = scene_dir_parts[-2]
        # scene_id should look like "zsNo4HB9uLZ"
        episode_id = str(env.current_episode.episode_id)
        # episode_id should look like "1475"

        print(f"[{episode_id} LOG] {scene_id}/{episode_id} reached")

        with open(SCENE_GRAPHS_PATH, 'rb') as f:
            scene_graphs = pickle.load(f)

        curr_sg = scene_graphs[scene_id]

        cached_actions_LOG = []
        who_acted_LOG = []
        did_cache_action_LOG = []
        cached_locations_LOG = []
        cached_yaws_LOG = []
        cached_directions_LOG = []
        early_stop_reason_LOG = []

        vlm_decision_count = 0
        continuse_rotation_count = 0
        last_dtg = None

        goal_pos = env.current_episode.goals[0].position
        goal_waypoint = find_nearest_waypoint_to(goal_pos, curr_sg)

        old_waypoint = None
        old_initial_orientation = None
        cached_temp_path = []

        # cached path is made up of multiple waypoints each with cached subpaths
        cached_path_in_progress = False
        cached_waypoints_to_visit = []
        cwtv_index = 0

        current_subpath_in_progress = False
        current_subpath_action_index = 0
        cached_action_list_to_follow = []
        cached_subpath_list_reverse = False
        desired_subpath_init_ori = None
        desired_ori_turning_required = False
        
        time_per_iter = []
        start_time = None
        end_time = None
        execution_time = None

        while not env.episode_over:
            info = env.get_metrics()
            if (start_time):
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                time_per_iter.append(execution_time)
                print(f"[{episode_id} LOG] Iteration performed in {execution_time}")
            else:
                print(f"[{episode_id} LOG] Starting iterations")

            start_time = time.perf_counter()

            curr_pos = env.sim.get_agent_state().position.tolist()
            curr_dir, curr_yaw = agent.get_cardinal_direction(env.sim.get_agent_state().rotation)

            who_acted = "cache"
            cached_this_iter = False

            previous_actor = who_acted_LOG[-1] if who_acted_LOG else None
            if previous_actor == "vlm":
                if last_dtg is None or info["distance_to_goal"] != last_dtg:
                    continuse_rotation_count = 0
                else:
                    continuse_rotation_count += 1
                last_dtg = info["distance_to_goal"]
            elif previous_actor is not None:
                continuse_rotation_count = 0
                last_dtg = None

            if (enable_use_of_cache):
                waypoint = find_nearest_waypoint_to(curr_pos, curr_sg, thresh=0.25)
                if (waypoint != None):
                    # try to cache the just finished path from previous point to this waypoint
                    if (enable_adding_to_cache):
                        if (old_waypoint) and (old_waypoint != waypoint):
                            if curr_sg.has_edge(old_waypoint, waypoint):
                                try:
                                    if (len(curr_sg[old_waypoint][waypoint]['cache_info']['cached_actions']) > cached_temp_path):
                                        # we have found a shorter path than our current cached path so we will replace it
                                        curr_sg[old_waypoint][waypoint]['cache_info'] = {'start': old_waypoint, 'start_ori': old_initial_orientation, 'cached_actions': cached_temp_path}
                                        cached_this_iter = True
                                except: # no cache info currently
                                    curr_sg[old_waypoint][waypoint]['cache_info'] = {'start': old_waypoint, 'start_ori': old_initial_orientation, 'cached_actions': cached_temp_path}
                                    cached_this_iter = True
                            else:
                                curr_sg.add_edge(old_waypoint, waypoint, cache_info={'start': old_waypoint, 'start_ori': old_initial_orientation, 'cached_actions': cached_temp_path})
                                cached_this_iter = True

                    old_waypoint = waypoint
                    old_initial_orientation = round(curr_yaw) # guaranteed to be one of 0, 30, 60, ..., 330 degrees
                    cached_temp_path = []
                    
                    # see if we can get on a cached path to the goal
                    if (not cached_path_in_progress):
                        cached_waypoints_to_visit = try_get_cached_path(waypoint, goal_waypoint, curr_sg)
                        if (cached_waypoints_to_visit): # FULL cached path exists
                            cached_action_list_to_follow = curr_sg[cached_waypoints_to_visit[0]][cached_waypoints_to_visit[1]]['cache_info']['cached_actions']
                            if (curr_sg[cached_waypoints_to_visit[0]][cached_waypoints_to_visit[1]]['cache_info']['start'] == waypoint):
                                cached_subpath_list_reverse = False
                                current_subpath_action_index = 0
                                desired_subpath_init_ori = curr_sg[cached_waypoints_to_visit[0]][cached_waypoints_to_visit[1]]['cache_info']['start_ori']
                            else:
                                cached_subpath_list_reverse = True
                                current_subpath_action_index = len(cached_action_list_to_follow) - 1
                                desired_subpath_init_ori = 180 - curr_sg[cached_waypoints_to_visit[0]][cached_waypoints_to_visit[1]]['cache_info']['start_ori']
                                if (desired_subpath_init_ori < 0):
                                    desired_subpath_init_ori += 360

                            cwtv_index = 1
                            current_subpath_in_progress = True
                            cached_path_in_progress = True
                            desired_ori_turning_required = True
                        else:
                            next_waypoint = find_best_cached_jump(curr_sg, waypoint, goal_waypoint)
                            if (next_waypoint):
                                cached_waypoints_to_visit = [] # empty list, we don't need it, will set it to this though so code
                                # that checks against it works properly
                                cwtv_index = 1 # placeholder value same as above
                                cached_action_list_to_follow = curr_sg[waypoint][next_waypoint]['cache_info']['cached_actions']
                                if (curr_sg[waypoint][next_waypoint]['cache_info']['start'] == waypoint):
                                    cached_subpath_list_reverse = False
                                    current_subpath_action_index = 0
                                    desired_subpath_init_ori = curr_sg[waypoint][next_waypoint]['cache_info']['start_ori']
                                else:
                                    cached_subpath_list_reverse = True
                                    current_subpath_action_index = len(cached_action_list_to_follow) - 1
                                    desired_subpath_init_ori = 180 - curr_sg[waypoint][next_waypoint]['cache_info']['start_ori']
                                    if (desired_subpath_init_ori < 0):
                                        desired_subpath_init_ori += 360
                                
                                current_subpath_in_progress = True
                                cached_path_in_progress = True
                                desired_ori_turning_required = True
                
                if (cached_path_in_progress):
                    if (not current_subpath_in_progress):
                        cwtv_index += 1
                        if (cwtv_index >= len(cached_waypoints_to_visit)):
                            cached_path_in_progress = False # destination reached
                        else:
                            cached_action_list_to_follow = curr_sg[cached_waypoints_to_visit[cwtv_index - 1]][cached_waypoints_to_visit[cwtv_index]]['cache_info']['cached_actions']
                            if (curr_sg[cached_waypoints_to_visit[cwtv_index - 1]][cached_waypoints_to_visit[cwtv_index]]['cache_info']['start'] == waypoint):
                                cached_subpath_list_reverse = False
                                current_subpath_action_index = 0
                                desired_subpath_init_ori = curr_sg[cached_waypoints_to_visit[cwtv_index - 1]][cached_waypoints_to_visit[cwtv_index]]['cache_info']['start_ori']
                            else:
                                cached_subpath_list_reverse = True
                                current_subpath_action_index = len(cached_action_list_to_follow) - 1
                                desired_subpath_init_ori = 180 - curr_sg[cached_waypoints_to_visit[cwtv_index - 1]][cached_waypoints_to_visit[cwtv_index]]['cache_info']['start_ori']
                                if (desired_subpath_init_ori < 0):
                                    desired_subpath_init_ori += 360
                            
                            current_subpath_in_progress = True
                            desired_ori_turning_required = True

                    if (not cached_path_in_progress): break

                    if (desired_ori_turning_required):
                        while (round(curr_yaw) > desired_subpath_init_ori):
                            desired_subpath_init_ori += 360
                        if (round(curr_yaw) == desired_subpath_init_ori):
                            desired_ori_turning_required = False
                        elif (desired_subpath_init_ori - round(curr_yaw) < 180):
                            # turn left
                            action = agent.act(obs, info, env.current_episode.episode_id, env, use_cached_action=True, cached_action=2)
                            continuse_rotation_count -= 1 # we are rotating a lot but we have a reason. avoid program flagging us
                        else: # the subtraction will be > 180 so turn right
                            action = agent.act(obs, info, env.current_episode.episode_id, env, use_cached_action=True, cached_action=3)
                            continuse_rotation_count -= 1

                    if (not desired_ori_turning_required):
                        if (cached_subpath_list_reverse):
                            cached_action_temp = cached_action_list_to_follow[current_subpath_action_index]
                            if (type(cached_action_temp) == dict):
                                cached_action_temp = cached_action_temp['action']
                            if (cached_action_temp == 2):
                                cached_action_temp = 3
                            elif (cached_action_temp == 3):
                                cached_action_temp = 2
                            action = agent.act(obs, info, env.current_episode.episode_id, env, use_cached_action=True, cached_action=cached_action_temp)
                            current_subpath_action_index -= 1
                            if (current_subpath_action_index < 0):
                                current_subpath_in_progress = False
                        else:
                            action = agent.act(obs, info, env.current_episode.episode_id, env, use_cached_action=True, cached_action=cached_action_list_to_follow[current_subpath_action_index])
                            current_subpath_action_index += 1
                            if (current_subpath_action_index >= len(cached_action_list_to_follow)):
                                current_subpath_in_progress = False
            
            if ((not enable_use_of_cache) or (not cached_path_in_progress)): # pure VLM navigation
                action = agent.act(obs, info, env.current_episode.episode_id, env)
                who_acted = getattr(agent, "last_action_source", "vlm")
                if getattr(agent, "last_action_is_vlm_decision", False):
                    vlm_decision_count += 1

            early_stop_reason = ""
            if who_acted == "vlm":
                if EARLY_STOP_ROTATION > 0 and continuse_rotation_count > EARLY_STOP_ROTATION:
                    early_stop_reason = f"rotation_stall>{EARLY_STOP_ROTATION}"
                elif EARLY_STOP_STEPS > 0 and vlm_decision_count > EARLY_STOP_STEPS:
                    early_stop_reason = f"vlm_decision_limit>{EARLY_STOP_STEPS}"
            if early_stop_reason:
                print(f"[{episode_id} LOG] Early stop triggered: {early_stop_reason}")
                action = {"action": 0}

            iter_step += 1
            obs = env.step(action)
            info = env.get_metrics()
            cached_temp_path.append(action)
            
            # FOR LOGGING
            #cached_locations_LOG.append(info['top_down_map_vlnce']['agent_map_coord']) # 2D MAP COORDS
            cached_locations_LOG.append(env.sim.get_agent_state().position.tolist()) # 3D HABITAT COORDS
            post_direction, post_yaw = agent.get_cardinal_direction(env.sim.get_agent_state().rotation)
            cached_yaws_LOG.append(float(post_yaw))
            cached_directions_LOG.append(str(post_direction))
            cached_actions_LOG.append(action['action'])
            who_acted_LOG.append(who_acted)
            did_cache_action_LOG.append(cached_this_iter)
            early_stop_reason_LOG.append(early_stop_reason)

            if (cached_this_iter):
                print(f"[{episode_id} LOG] Added cached path to (local) scene graph")
            if (cached_path_in_progress):
                print(f"[{episode_id} LOG] Executing cache path...")

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        time_per_iter.append(execution_time)

        info = env.get_metrics()
        result_dict = dict()
        result_dict = {k: info[k] for k in target_key if k in info}
        result_dict["id"] = env.current_episode.episode_id
        count += 1

        with open(
            os.path.join(
                os.path.join(result_path, "log"),
                "stats_{}.json".format(env.current_episode.episode_id),
            ),
            "w",
        ) as f:
            json.dump(result_dict, f, indent=4)

        # consider to be goal waypoint if success
        if (enable_adding_to_cache):
            if (result_dict["success"]):
                cached_this_iter = False
                if (old_waypoint):
                    if curr_sg.has_edge(old_waypoint, goal_waypoint):
                        try:
                            if (len(curr_sg[old_waypoint][goal_waypoint]['cache_info']['cached_actions']) > cached_temp_path):
                                # we have found a shorter path than our current cached path so we will replace it
                                curr_sg[old_waypoint][goal_waypoint]['cache_info'] = {'start': old_waypoint, 'start_ori': old_initial_orientation, 'cached_actions': cached_temp_path}
                                cached_this_iter = True
                        except: # no cache info currently
                            curr_sg[old_waypoint][goal_waypoint]['cache_info'] = {'start': old_waypoint, 'start_ori': old_initial_orientation, 'cached_actions': cached_temp_path}
                            cached_this_iter = True
                    else:
                        curr_sg.add_edge(old_waypoint, goal_waypoint, cache_info={'start': old_waypoint, 'start_ori': old_initial_orientation, 'cached_actions': cached_temp_path})
                        cached_this_iter = True

                scene_graphs[scene_id] = curr_sg

                print(f"[{episode_id} LOG] Successful outcome. Writing cached paths to file")

                with open(SCENE_GRAPHS_PATH, 'wb') as f:
                    pickle.dump(scene_graphs, f)

        ssa_trace_path = ""
        if getattr(agent, "ssa_controller", None):
            ssa_trace_path = agent.ssa_controller.save_episode_trace(result_path, episode_id)
        cache_dict = {
            "scene_id": scene_id,
            "success": result_dict["success"],
            "avg_time_per_iter": statistics.mean(time_per_iter) if time_per_iter else 0,
            "num_vlm_calls": who_acted_LOG.count("vlm"),
            "avg_price_per_vlm_call": statistics.mean(agent.total_costs_of_calls) if agent.total_costs_of_calls else 0,
            "num_total_calls": len(who_acted_LOG),
            "goal_waypoint": goal_waypoint,
            "cached_actions_LOG": cached_actions_LOG,
            "who_acted_LOG": who_acted_LOG,
            "did_cache_action_LOG": did_cache_action_LOG,
            "cached_locations_LOG": cached_locations_LOG,
            "cached_yaws_LOG": cached_yaws_LOG,
            "cached_directions_LOG": cached_directions_LOG,
            "early_stop_reason_LOG": early_stop_reason_LOG,
            "vlm_price_LOG": agent.total_costs_of_calls,
            "ssa_takeover_used": bool(getattr(agent.ssa_controller, "used_this_episode", False)) if getattr(agent, "ssa_controller", None) else False,
            "ssa_summary": agent.ssa_controller.episode_summary() if getattr(agent, "ssa_controller", None) else {},
            "ssa_trace_path": ssa_trace_path,
        }
        if getattr(agent, "ssa_controller", None):
            ssa_trace = agent.ssa_controller.episode_trace()
            ssa_trajectory = []
            for takeover_result in ssa_trace.get("takeover_results", []) or []:
                ssa_trajectory.extend(takeover_result.get("ssa_trajectory", []) or [])
            goal_position = []
            try:
                goal_position = env.current_episode.goals[0].position
            except Exception:
                goal_position = []
            save_trajectory_debug(
                output_dir=result_path,
                episode_id=str(env.current_episode.episode_id),
                payload={
                    "episode_id": str(env.current_episode.episode_id),
                    "scene_id": scene_id,
                    "metric": result_dict,
                    "start_position": env.current_episode.start_position,
                    "goal_position": goal_position,
                    "agent_trajectory": [
                        {
                            "step": int(i),
                            "source": str(who_acted_LOG[i]) if i < len(who_acted_LOG) else "agent",
                            "position": pos,
                            "yaw_deg": cached_yaws_LOG[i] if i < len(cached_yaws_LOG) else None,
                            "action": cached_actions_LOG[i] if i < len(cached_actions_LOG) else None,
                        }
                        for i, pos in enumerate(cached_locations_LOG)
                    ],
                    "ssa_trajectory": ssa_trajectory,
                    "expert_trajectory": [],
                    "expert_trajectory_available": False,
                    "ssa_trace": ssa_trace,
                },
            )

        if getattr(agent, "ssa_controller", None):
            print(f"[SSA] episode summary | episode={episode_id} {agent.ssa_controller.episode_summary_text()}")
        print(format_episode_metric(episode_id, result_dict))

        with open(os.path.join(os.path.join(result_path, "cache_log"),"stats_{}.json".format(env.current_episode.episode_id)), "w") as f:
            json.dump(cache_dict, f, indent=4)
        agent.save_episode_gif()

        if (result_dict["success"]):
            print(f"[{episode_id} LOG] {scene_id}/{episode_id} complete, SUCCESS")
        else:
            print(f"[{episode_id} LOG] {scene_id}/{episode_id} complete, FAILURE")



class MyGPTAgent(Agent):
    def __init__(
        self,
        result_path,
        require_map=True,
        workspace_root=WORKSPACE_ROOT,
        ssa_guidance=False,
        ssa_checkpoint="",
        ssa_detect_threshold=0.30,
        ssa_detector_model_source="",
        filter_behind=False,
        oracle_exit_enable=False,
    ):
        # print("Initialize MyGPTAgent")

        self.result_path = result_path
        self.require_map = require_map
        self.workspace_root = workspace_root

        self.total_costs_of_calls = []

        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "video"), exist_ok=True)
        os.makedirs(os.path.join(self.result_path, "cache_log"), exist_ok=True)

        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL", "https://models.sjtu.edu.cn/api/v1"))
        print(f"DEBUG: client base_url={self.client.base_url}")

        # Initialize tracking variables
        self.rgb_list = []
        self.topdown_map_list = []
        self.count_id = 0
        self.previous_plan = None
        self.previous_output = None
        self.step_count = 0
        self.history = []
        self.history_window = 5
        self.last_action = None
        self.ssa_controller = SSAController(
            enabled=bool(ssa_guidance),
            workspace_root=Path(self.workspace_root),
            checkpoint_path=ssa_checkpoint,
            detect_threshold=float(ssa_detect_threshold),
            detector_model_source=ssa_detector_model_source or None,
            filter_behind=filter_behind,
            oracle_exit_enabled=oracle_exit_enable,
        )

        # print("Initialization Complete")

        self.reset()

    def encode_image(self, image_array):
        buffered = io.BytesIO()
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:
                img = Image.fromarray(image_array, mode="RGBA")
            else:
                img = Image.fromarray(image_array, mode="RGB")
        else:
            img = Image.fromarray(image_array)

        max_dim = 320
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _ssa_infer(self, system_prompt: str, user_prompt: str) -> str:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
        request_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": int(os.getenv("SSA_DELEGATE_MAX_TOKENS", "32")),
            "temperature": 0,
        }
        extra_body = self.build_chat_extra_body(model_name)
        if extra_body:
            request_params["extra_body"] = extra_body
        response = self.client.chat.completions.create(**request_params)
        if getattr(response, "usage", None) is not None:
            self.total_costs_of_calls.append(openai_api_calculate_cost(response.usage))
        return self.extract_llm_text(response.choices[0].message)

    def _ssa_infer_with_images(self, system_prompt: str, user_prompt: str, images: dict) -> str:
        model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
        user_content = [{"type": "text", "text": user_prompt}]
        for image_dict in (images or {}).values():
            rgb = image_dict.get("rgb") if isinstance(image_dict, dict) else image_dict
            image_url = self._ssa_image_data_url(rgb)
            if image_url:
                user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        request_params = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": int(os.getenv("SSA_DELEGATE_MAX_TOKENS", "32")),
            "temperature": 0,
        }
        extra_body = self.build_chat_extra_body(model_name)
        if extra_body:
            request_params["extra_body"] = extra_body
        response = self.client.chat.completions.create(**request_params)
        if getattr(response, "usage", None) is not None:
            self.total_costs_of_calls.append(openai_api_calculate_cost(response.usage))
        return self.extract_llm_text(response.choices[0].message)

    def _ssa_image_data_url(self, rgb) -> str:
        if rgb is None:
            return ""
        if isinstance(rgb, Image.Image):
            arr = np.asarray(rgb.convert("RGB"))
        else:
            arr = np.asarray(rgb)
        if arr.dtype != np.uint8 and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[-1] > 3:
            arr = arr[..., :3]
        return f"data:image/png;base64,{self.encode_image(arr)}"

    def get_cardinal_direction(self, quaternion):
        """
        Return 16-point cardinal label and *clockwise* yaw in degrees
        (0° = North, 90° = East, 180° = South, 270° = West).
        """
        heading_vector = quaternion_rotate_vector(
            quaternion.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        yaw = np.rad2deg(np.array(phi) + z_neg_z_flip)

        dirs = [
            "S",
            "SSE",
            "SE",
            "ESE",
            "E",
            "ENE",
            "NE",
            "NNE",
            "N",
            "NNW",
            "NW",
            "WNW",
            "W",
            "WSW",
            "SW",
            "SSW",
        ]
        idx = round(yaw / 24)
        return dirs[idx], yaw

    def parse_action_number(self, response_text):
        action_match = re.search(r"Action:\s*\[?(\d+)\]?", response_text, re.IGNORECASE)
        if action_match:
            action_num = int(action_match.group(1))
            if 0 <= action_num <= 3:
                return action_num

        try:
            first_char = response_text.strip()[0]
            if first_char.isdigit():
                action_num = int(first_char)
                if 0 <= action_num <= 3:
                    return action_num
        except (ValueError, IndexError):
            pass

        first_line = response_text.split("\n")[0]
        for char in first_line:
            if char in "0123":
                action_num = int(char)
                return action_num

        print("No valid action found, defaulting to 0 (stop)")
        return 0

    def parse_next_step(self, generated_text: str) -> str:
        match = re.search(
            r"Next step:\s*(.+)", generated_text, re.IGNORECASE | re.DOTALL
        )
        if match:
            next_step = match.group(1).strip()
            next_step = next_step.split("\n")[0].strip()
            return next_step
        else:
            return ""

    def parse_ssa_delegate(self, generated_text: str) -> bool:
        match = re.search(
            r"Delegate to SSA:\s*(yes|no|true|false)",
            generated_text,
            re.IGNORECASE,
        )
        if not match:
            return False
        return match.group(1).strip().lower() in {"yes", "true"}

    def extract_llm_text(self, message) -> str:
        """Support thinking-model responses where final content may be None."""
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            joined = "\n".join(part.strip() for part in parts if part and part.strip())
            if joined:
                return joined
        for attr in ("reasoning_content", "reasoning", "thinking"):
            value = getattr(message, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if hasattr(message, "model_dump"):
            dumped = message.model_dump()
            for key in ("content", "reasoning_content", "reasoning", "thinking"):
                value = dumped.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    def build_chat_extra_body(self, model: str) -> dict:
        """Disable Qwen thinking by default; navigation only needs short actions."""
        model_name = str(model or "").lower()
        if "qwen" not in model_name:
            return {}
        enable_thinking = os.getenv("QWEN_ENABLE_THINKING", "0").strip().lower()
        if enable_thinking in {"1", "true", "yes", "on"}:
            return {}
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def get_topdown_map_base64(self, info, rgb_shape):
        if "top_down_map_vlnce" in info:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb_shape[0]
            )

            if top_down_map.dtype != np.uint8:
                top_down_map = (top_down_map * 255).astype(np.uint8)

            plt.figure(figsize=(8, 8))
            plt.imshow(top_down_map)
            plt.title("Top-Down Map")
            plt.axis("off")
            buf = io.BytesIO()
            # plt.savefig(
            #     "tmp/testing/top_down_map.png",
            #     format="png",
            #     bbox_inches="tight",
            #     pad_inches=0,
            # )
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)

            img_bytes = buf.getvalue()
            map_img_str = base64.b64encode(img_bytes).decode("utf-8")

            plt.close()

            return map_img_str
        return None

    def get_simple_topdown_map_base64(self, env):
        agent_state = env.sim.get_agent_state()
        goal_pos = env.current_episode.goals[0].position
        top_down_map = maps.get_topdown_map_from_sim(env.sim)
        recolor_map = np.array(
            [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
        )
        top_down_map = recolor_map[top_down_map]
        # Convert from Habitat's coordinate system to matplotlib's
        coords = maps.to_grid(
            agent_state.position[2],
            agent_state.position[0],
            (top_down_map.shape[0], top_down_map.shape[1]),
            sim=env.sim,
        )
        # Convert from quaternion to yaw angle
        rot = agent_state.rotation
        yaw = np.pi + np.arctan2(
            2 * rot.y * rot.w - 2 * rot.x * rot.z,
            1 - 2 * rot.y * rot.y - 2 * rot.z * rot.z,
        )
        # Add marker for start agent position
        agent_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=coords,
            agent_rotation=yaw,
            agent_radius_px=50,
        )

        plt.imshow(agent_map)
        plt.title("Annotated Top-Down Map")
        plt.grid(False)
        plt.axis("off")
        plt.savefig(
            "tmp/testing/top_down_map.png",
            format="png",
            bbox_inches="tight",
            pad_inches=0,
        )
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        goal_plt_pos = list(
            maps.to_grid(
                goal_pos[2],
                goal_pos[0],
                (top_down_map.shape[0], top_down_map.shape[1]),
                sim=env.sim,
            )
        )
        plt.scatter(goal_plt_pos[1], goal_plt_pos[0], c="yellow", s=30)
        plt.text(
            goal_plt_pos[1],
            goal_plt_pos[0] - 60,
            "Goal",
            fontsize=7,
            color="black",
            ha="center",
            va="top",
        )
        img_bytes = buf.getvalue()
        map_img_str = base64.b64encode(img_bytes).decode("utf-8")

        return map_img_str


    def addtext(self, image, instruction, navigation, current_direction, yaw):
        """Add text overlay to image with wrapping and shrink-to-fit"""
        h, w = image.shape[:2]
        new_height = h + 200
        new_image = np.ones((new_height, w, 3), np.uint8) * 255
        new_image[:h, :w] = image

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_spacing = 5

        # Build text
        all_text = f"Current direction: {current_direction}. {instruction}"
        lines = []
        words = all_text.split(" ")
        line = ""

        for word in words:
            test_line = (line + " " + word).strip()
            size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            if size[0] > w - 20:  # wrap if too wide
                lines.append(line)
                line = word
            else:
                line = test_line
        if line:
            lines.append(line)

        # Add navigation on a new line
        if navigation == None:
            navigation = "No navigation reasoning provided."
        nav_words = navigation.split(" ")
        nav_line = ""
        for word in nav_words:
            test_line = (nav_line + " " + word).strip()
            size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            if size[0] > w - 20:
                lines.append(nav_line)
                nav_line = word
            else:
                nav_line = test_line
        if nav_line:
            lines.append(nav_line)

        # Adjust font size if still overflowing vertically
        while True:
            text_height = len(lines) * (
                cv2.getTextSize("Test", font, font_scale, thickness)[0][1] + line_spacing
            )
            if h + text_height < new_height:
                break
            font_scale -= 0.05
            if font_scale < 0.3:
                break

        # Render text
        y = h + 30
        for line in lines:
            cv2.putText(new_image, line, (10, y), font, font_scale, (0, 0, 0), thickness)
            y += int(
                cv2.getTextSize(line, font, font_scale, thickness)[0][1] + line_spacing
            )

        return new_image

    def reset(self):
        self.save_episode_gif()

        self.rgb_list = []
        self.topdown_map_list = []
        self.previous_plan = None
        self.previous_output = None
        self.step_count = 0
        self.count_id += 1
        self.pending_action_list = []
        self.total_costs_of_calls = []
        self.ssa_controller.reset()
        self.pending_ssa_prediction = None
        self.pending_ssa_before_position = None
        self.last_action_source = "vlm"
        self.last_action_is_vlm_decision = False
        self._cached_instruction = None

    def save_episode_gif(self):
        if not self.require_map or len(self.topdown_map_list) == 0:
            return
        output_video_path = os.path.join(
            self.result_path, "video", "{}.gif".format(self.episode_id)
        )
        imageio.mimsave(output_video_path, self.topdown_map_list)

    def _ssa_current_position(self, env):
        return [float(v) for v in env.sim.get_agent_state().position.tolist()]

    @staticmethod
    def _ssa_action_to_id(action_name):
        return {"FORWARD": 1, "LEFT": 2, "RIGHT": 3}.get(str(action_name), 1)

    def _ssa_record_previous_action_result(self, env, collision: bool):
        if self.pending_ssa_prediction is None:
            return
        self.ssa_controller.record_action_result(
            self.pending_ssa_prediction,
            before_position=self.pending_ssa_before_position,
            after_position=self._ssa_current_position(env),
            collision=bool(collision),
            done=False,
        )
        self.pending_ssa_prediction = None
        self.pending_ssa_before_position = None

    def _ssa_next_action(self, rgb, depth, env, instruction, info, current_direction, current_yaw):
        prediction = self.ssa_controller.predict_step(
            rgb=rgb,
            depth=depth,
            current_position=self._ssa_current_position(env),
        )
        if prediction.get("exit", False):
            self.previous_plan = "SSA stair takeover finished."
            self.previous_output = f"SSA takeover finished: {prediction.get('exit_reason', '')}"
            return None

        action_id = self._ssa_action_to_id(prediction.get("action", "FORWARD"))
        self.pending_ssa_prediction = prediction
        self.pending_ssa_before_position = self._ssa_current_position(env)
        self.previous_plan = "SSA stair takeover in progress."
        self.previous_output = (
            f"SSA phase={prediction.get('phase', '')}; "
            f"action={prediction.get('action', '')}"
        )
        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
            output_im = np.concatenate((rgb, top_down_map), axis=1)
            img = self.addtext(
                output_im,
                instruction,
                self.previous_output,
                current_direction,
                current_yaw,
            )
            self.topdown_map_list.append(img)
        self.last_action = action_id
        self.last_action_source = "ssa"
        return {"action": action_id}

    def _get_instruction(self, observations, env):
        if self._cached_instruction is not None:
            return self._cached_instruction
        inst = observations.get("instruction")
        if isinstance(inst, dict):
            text = inst.get("text", "")
        else:
            text = ""
        if not text and hasattr(env, "current_episode"):
            text = getattr(env.current_episode.instruction, "text", "")
        if not text:
            text = "<instruction not available>"
        self._cached_instruction = text
        return text

    def act(self, observations, info, episode_id, env, use_cached_action=False, cached_action=None, end_scene_on_cached_0=False):
        self.episode_id = episode_id
        self.step_count += 1
        self.last_action_is_vlm_decision = False
        rgb = observations["rgb"]
        self.rgb_list.append(rgb)
        agent_state = env.sim.get_agent_state()
        if agent_state is not None:
            current_direction, current_yaw = self.get_cardinal_direction(
                agent_state.rotation
            )
        else:
            raise ValueError("Agent state not available")

        if (
            len(self.pending_action_list) != 0
        ):  # Pending action queue so GPT isn't queried every step
            temp_action = self.pending_action_list.pop(
                0
            )  # Run steps in queue before requerying gpt
            self.last_action_source = "vlm"

            # if self.require_map:
            #     top_down_map = maps.colorize_draw_agent_and_fit_to_height(
            #         info["top_down_map_vlnce"], rgb.shape[0]
            #     )
            #     output_im = np.concatenate((rgb, top_down_map), axis=1)
            #     img = self.addtext(
            #         output_im,
            #         observations["instruction"]["text"],
            #         f"Pending action: {temp_action}",
            #         current_direction,
            #         current_yaw
            #     )
            #     self.topdown_map_list.append(img)

            return {"action": temp_action}

        if self.step_count % 1 == 0:
            instruction = self._get_instruction(observations, env)

            # collision = observations.get("collisions", {}).get("is_collision", False)
            collision_info = info.get(
                "collisions", 0
            )  # Added a collision measurement for agent's decision making
            collision = (
                collision_info.get("is_collision", False)
                if isinstance(collision_info, dict)
                else False
            )  # returns T or F if in collision
            self._ssa_record_previous_action_result(env, bool(collision))
            if self.ssa_controller.takeover_active:
                ssa_action = self._ssa_next_action(
                    rgb,
                    observations.get("depth"),
                    env,
                    instruction,
                    info,
                    current_direction,
                    current_yaw,
                )
                if ssa_action is not None:
                    return ssa_action

            self.history.append(
                {
                    "step": self.step_count,
                    "action": getattr(self, "last_action", None),
                    "direction": current_direction,
                    "yaw": current_yaw,
                    "collision": collision,
                }
            )
            self.history = self.history[-self.history_window :]

            history_text = "Recent actions (last 5):\n"  # Summarize last 5 actions for better decision making
            for h in self.history:
                if h["action"] is None:
                    continue
                hist_line = (
                    f"Step {h['step']}: Action={h['action']} | "
                    f"Dir={h['direction']} | Yaw={h['yaw']:.1f} | "
                    f"Collision={h['collision']}"
                )
                history_text += hist_line + "\n"

            if len(self.history) >= 4:
                last_actions = [
                    h["action"] for h in self.history if h["action"] is not None
                ]
                if len(last_actions) >= 4:
                    if sum(a in [2, 3] for a in last_actions) == len(last_actions):
                        history_text += "\n⚠️ Warning: Loop detected (rotating in place).\n"  # If repeated turns, agent is stuck in loop

            map_img_str = self.get_topdown_map_base64(info, rgb.shape)
            image_data = self.encode_image(rgb)
            ssa_proposal = self.ssa_controller.update_proposal(
                instruction=instruction,
                previous_output=self.previous_output or "",
                previous_plan=self.previous_plan or "",
                rgb=rgb,
                depth=observations.get("depth"),
            )
            self.ssa_controller.record_step_proposal(
                step=self.step_count,
                available=bool(ssa_proposal.get("available", False)),
                reason=str(ssa_proposal.get("reason", "")),
            )
            print(
                f"[SSA] step={self.step_count} episode={episode_id} "
                f"available={bool(ssa_proposal.get('available', False))} "
                f"reason={ssa_proposal.get('reason', '')}"
            )
            # map_img_str = self.get_topdown_map_base64(info, rgb.shape)
            user_text = (
                f"Navigate to approach the red square on the top down map using the top-down map and camera view.\n\n"
                f"TASK INSTRUCTION: Navigate until the agent's arrow is on top of the red square on the top down map.\n\n"
                f"AGENT ORIENTATION:\n"
                f"- Current cardinal direction: {current_direction}\n"
                f"- Yaw angle: {current_yaw:.1f}°\n\n"
            )

            distance_to_goal = info.get("distance_to_goal", None)
            if distance_to_goal is not None:
                user_text += f"\nDISTANCE TO GOAL: {distance_to_goal:.2f} meters\n\n"

            if self.previous_plan:
                user_text += f"Plans from previous step:\n{self.previous_plan}\n\n"

            if history_text:
                user_text += "=== HISTORY CONTEXT ===\n" + history_text + "\n"

            user_text += (
                "=== SSA STAIR TAKEOVER ===\n"
                f"SSA available: {'True' if ssa_proposal.get('available', False) else 'False'}\n"
            )

            # Output options for actions
            user_text += (
                f"AVAILABLE ACTIONS:\n"
                f"0) Stop (task complete)\n"
                f"1) Move forward\n"
                f"2) Turn left\n"
                f"3) Turn right\n\n"
                f"Analyze the image and plan your next move using **global cardinal directions**."
            )

            messages = [  # System prompt for navigation
                {
                    "role": "system",
                    "content": (
                        "You are an AI navigation agent inside a simulated environment. "
                        "Your job is to move from your START location to a GOAL location using your top-down map and camera view.\n\n"
                        "=== MAP LEGEND ===\n"
                        "- BLUE SQUARE: Your starting position.\n"
                        "- BLUE ARROW: Your current position & facing direction.\n"
                        "- BLUE LINE: Your path so far.\n"
                        "- RED SQUARE: The goal location you must reach.\n"
                        "- GRAY AREAS: Navigable floor where you can walk.\n"
                        "- WHITE AREAS: Obstacles or walls you cannot walk through.\n\n"
                        "=== NAVIGATION PRINCIPLES ===\n"
                        "1. Always identify the RED SQUARE (goal) on the map.\n"
                        "2. Compare your CURRENT CARDINAL DIRECTION (N, NE, E, SE, S, SW, W, NW) with the DIRECTION from your location to the goal.\n"
                        "3. If not facing toward the goal, turn left (2) or right (3) to align your heading.\n"
                        "4. Move forward (1) only when facing an open navigable path toward the goal using the FRONT VIEW CAMERA.\n"
                        "5. Avoid white (non-navigable) areas — if blocked, reorient using the map.\n"
                        "6. Stop (0) when you reach the goal on the top-down map (ONLY when blue arrow is touching or on top of red square).\n\n"
                        "7. Use distance-to-goal as feedback: if it's close to zero, you are near the goal, and you should Stop(0).\n"
                        "If it increases or stays constant for several steps, adjust strategy.\n\n"

                        "=== DECISION RULES ===\n"
                        "- Use top-down map to determine direction to move in.\n"
                        "- Every step, make a micro-plan: identify goal direction, check navigability, choose turn/move.\n"
                        "- If goal is to your left/right on the map, rotate toward it before moving.\n"
                        "- Use global cardinal directions for reasoning, NOT relative left/right from the camera.\n\n"
                        # "=== INSTRUCTION HANDLING PRINCIPLES ==="
                        # "- Align map reasoning with the goal (if the instruction says 'enter the room on the left,' prioritize detecting a doorway on the left side of the map/camera).\n"
                        # "- If the map doesn’t directly show the described feature (e.g., 'hallway'), rely on camera view and relative navigation until a landmark matches.\n"
                        "=== HISTORY INTERPRETATION RULES ==="
                        "- If Collision=True, treat it as a collision (forward failed) → reroute using a different action.\n"
                        "- If the last few actions are all turns (2 or 3) and orientation is nearly unchanged, you are looping → choose a different strategy (try forward or opposite turn).\n"
                        "- Use history and past path to avoid repeating the same failed action sequence.\n"
                        "=== SSA STAIR TAKEOVER ===\n"
                        "- SSA is a dedicated local stair traversal controller.\n"
                        "- Do not decide whether the whole route needs stairs only from the full instruction.\n"
                        "- If `SSA available: True`, output `Delegate to SSA: Yes` when your current local next step is to approach, enter, traverse, go up, go down, or leave stairs.\n"
                        "- Being already on stairs is not a reason to reject SSA if the current local task is still stair traversal.\n"
                        "- If you delegate to SSA, it will take over local low-level actions and later return control.\n"
                        "- If `SSA available: False`, output `Delegate to SSA: No`.\n"
                        "=== OUTPUT FORMAT ===\n"
                        "Delegate to SSA: [Yes/No]\n"
                        "Action: [0-3]\n"
                        "Map reasoning: [Describe goal location relative to you in cardinal terms]\n"
                        "Camera reasoning: [Objects / obstacles seen in current view]\n"
                        "Navigation reasoning: [Step-by-step plan using map + camera]\n"
                        "Next step: [Brief plan for next move]"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{map_img_str}"
                            },
                        },
                    ],
                },
            ]
            try:
                if (use_cached_action == False):
                    max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "4096"))
                    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1")
                    extra_body = self.build_chat_extra_body(model_name)
                    print(f"[LLM] request max_tokens={max_tokens} model={model_name} extra_body={extra_body}")
                    request_params = {
                        "model": model_name,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.3,
                    }
                    if extra_body:
                        request_params["extra_body"] = extra_body
                    response = self.client.chat.completions.create(**request_params)

                    print(response)

                    generated_text = self.extract_llm_text(response.choices[0].message)
                    if not generated_text:
                        generated_text = (
                            "Delegate to SSA: No\n"
                            "Action: 1\n"
                            "Next step: Empty LLM response; continue cautiously."
                        )
                        print("[LLM] empty response content; using safe fallback action=1")
                    self.last_action_is_vlm_decision = True
                    self.total_costs_of_calls.append(openai_api_calculate_cost(response.usage))
                    self.previous_plan = self.parse_next_step(generated_text)
                    self.previous_output = generated_text
                    ssa_after_response = self.ssa_controller.update_proposal(
                        instruction="",
                        previous_output=generated_text,
                        previous_plan=self.previous_plan or "",
                        rgb=rgb,
                        depth=observations.get("depth"),
                        delegate_infer_fn=self._ssa_infer,
                        delegate_image_infer_fn=self._ssa_infer_with_images,
                        delegate_image={"rgb": rgb},
                        delegate_current_stage=self.previous_plan or generated_text,
                        delegate_history=history_text,
                        delegate_observation_hint=(
                            f"distance_to_goal={info.get('distance_to_goal', 'unknown')}"
                        ),
                    )
                    if ssa_after_response.get("available", False):
                        ssa_direction = str(ssa_after_response.get("direction", "unknown"))
                        delegate_info = ssa_after_response.get("delegate", {}) if isinstance(ssa_after_response.get("delegate"), dict) else {}
                        delegate_reason = "vlm_fallback" if ssa_after_response.get("reason") == "delegate_vlm" else "rule_and_dino_gate"
                        self.ssa_controller.record_delegate_decision(
                            step=self.step_count,
                            delegated=True,
                            current_stage=self.previous_plan or "",
                            history=history_text,
                            prompt_has_rgb=bool(delegate_info.get("prompt_has_rgb", False)),
                            raw_response=str(delegate_info.get("raw_response", "") or generated_text),
                            reason=delegate_reason,
                            direction=ssa_direction,
                        )
                        self.ssa_controller.record_plan_outcome(
                            step=self.step_count,
                            accepted=True,
                            reason="closed_loop_ready",
                            planned_actions=0,
                        )
                        self.ssa_controller.start_takeover(
                            direction=ssa_direction,
                            step=self.step_count,
                            rgb=rgb,
                            depth=observations.get("depth"),
                            oracle_exit=select_oracle_exit_for_episode(
                                env.current_episode,
                                current_position=self._ssa_current_position(env),
                                direction=ssa_direction,
                            ),
                        )
                        print(f"[SSA] step={self.step_count} episode={episode_id} delegated=yes mode=closed_loop direction={ssa_direction}")
                        ssa_action = self._ssa_next_action(
                            rgb,
                            observations.get("depth"),
                            env,
                            instruction,
                            info,
                            current_direction,
                            current_yaw,
                        )
                        if ssa_action is not None:
                            return ssa_action
                    action_index = self.parse_action_number(generated_text)
                else:
                    if (type(cached_action) == dict):
                        cached_action = cached_action['action']
                    if ((cached_action == 0) and (not end_scene_on_cached_0)): # avoid stopping early if cache is intermediate path, unless it's intended of course
                        cached_action = 1
                    response = f"Action: {cached_action}\nNext step: Move to waypoint along known path."
                    generated_text = response
                    self.previous_plan = "Move to waypoint along known path."
                    self.previous_output = generated_text
                    action_index = cached_action

                # print(f"\nModel decision: {action_index}\n")
                # 0 is stop
                # 1 is move forward
                # 2 is turn left
                # 3 is turn right
                
                # We avoid appending multiple low level actions when using cache;
                # this INCREASES gpt calls but also INCREASES reliability of cache working.
                # It is recommended to use SINGLE-ACTION APPEND if using the cache.
                # AVOID SWITCHING BETWEEN MULTI-ACTION AND SINGLE-ACTION WHEN USING CACHE DATA!

                # MULTI-ACTION APPEND (cheap, potentially harmful interactions with cache)
                # if action_index == 0:
                #     self.pending_action_list.append(0)
                # elif action_index == 1:
                #     for _ in range(
                #         3
                #     ):  # We add multiple low level actions to prevent GPT calls every step
                #         self.pending_action_list.append(1)
                # elif action_index == 2:
                #     for _ in range(2):
                #         self.pending_action_list.append(2)
                # elif action_index == 3:
                #     for _ in range(2):
                #         self.pending_action_list.append(3)

                # SINGLE-ACTION APPEND (more VLM calls, more accurate cache)
                if (action_index in [0, 1, 2, 3]):
                    self.pending_action_list.append(action_index)
                    self.last_action_source = "cache" if use_cached_action else "vlm"
                    self.last_action_is_vlm_decision = not use_cached_action

            except Exception as e:
                print(f"API Error: {e}")
                traceback.print_exc()
                self.pending_action_list.append(random.randint(1, 3))
                self.last_action_source = "fallback"
        else:
            # self.pending_action_list.append(1)
            pass

        if len(self.pending_action_list) == 0:  # Start with a stop sig
            self.pending_action_list.append(0)
            self.last_action_source = "fallback_empty"

        if self.require_map:
            top_down_map = maps.colorize_draw_agent_and_fit_to_height(
                info["top_down_map_vlnce"], rgb.shape[0]
            )
            output_im = np.concatenate((rgb, top_down_map), axis=1)

            action_text = f"Next action: {self.pending_action_list[0]}"
            if hasattr(self, "previous_plan") and self.previous_plan:
                action_text += f" | Plan: {self.previous_plan[:50]}..."

            img = self.addtext(
                output_im,
                self._get_instruction(observations, env),
                self.previous_output,
                current_direction,
                current_yaw,
            )
            self.topdown_map_list.append(img)

        self.last_action = self.pending_action_list[0]

        return {"action": self.pending_action_list.pop(0)}
