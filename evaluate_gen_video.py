import argparse
import os
import ray
from ray import tune
from current_config_xy import get_config
from ray.rllib.agents.ppo import PPOTrainer
from Coop_env_call_out import MultiAgentSync_call,MultiAgentSing_call
from MultiAgentSync_fullobs_samefield_randnose_xycoords import (
    MultiAgentSync_fullobs,
    MultiAgentSing_fullobs,
    MultiAgentSync_noobs,
    MultiAgentSing_noobs,
)
from Coop_env_call_to_observe import MultiAgentSync_call2,MultiAgentSing_call2
from Coop_env_memory_old_ import MultiAgentSing_memory,MultiAgentSync_memory
import imageio
import numpy as np


def get_args():
    import torch

    parser = argparse.ArgumentParser(description="Train MARL cooperation agents")

    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="which gpu to use - 0 or 1",
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.3,
        help="Number of GPUs to run on (can be a fraction)",
    )
    parser.add_argument(
        "--env_dir",
        type=str,
        default="MultiAgentSync_fullobs_samefield_randnose_xycoords.py",
        help="environment file",
    )
    parser.add_argument(
        "--condition",
        type=str,
        choices=[
            "MultiAgentSync_fullobs",
            "MultiAgentSing_fullobs",
            "MultiAgentSync_noobs",
            "MultiAgentSing_noobs",
            "MultiAgentSync_call",
            "MultiAgentSing_call",
            "MultiAgentSync_call2",
            "MultiAgentSing_call2",
            "MultiAgentSing_memory",
            "MultiAgentSync_memory"
        ],
        default="MultiAgentSync_fullobs",
        help="condition choices: non-coop/coop, fullobs/noobs",
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="training folder",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="./videos",
    )
    parser.add_argument(
        "--l2_curr", type=float, default=0.01, help="L2-reg on recurrence"
    )

    parser.add_argument("--l2_inp", type=float, default=0, help="L2-reg on input")
    parser.add_argument(
        "--coop_window", type=int, default=2, help="time window for agents cooperation"
    )
    parser.add_argument(
        "--fix_loc",
        action="store_true",  # This sets it to true if the flag is provided
        help="Disable randomizing np and water after correct (default: False)",
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",  # This sets it to true if the flag is provided
        help="pretrain stage with all observations provided",
    )
    parser.add_argument(
        "--randomize_miss",
        type=bool,
        default=False,
        help="randomize np and water after miss, generally false",
    )
    args = parser.parse_args()

    if args.num_gpus > 0 and not torch.cuda.is_available():
        print("No GPU available. Setting num_gpus to 0.")
        args.num_gpus = 0  # Use CPU if no GPU is available

    print("Running trails with the following arguments: ", args)
    return args


def save_video(image_list, output_path="output.mp4", fps=2):

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the images to an MP4 video
    with imageio.get_writer(output_path, format="FFMPEG", fps=fps) as writer:
        for image in image_list:
            writer.append_data(np.array(image))


def find_best_iteration(ck_path):
    ck_path = os.path.abspath(ck_path)
    # print(f"Received path: {ck_path}")
    analysis = tune.ExperimentAnalysis(ck_path)
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean",mode="max"),
        metric="episode_reward_mean",
    )
    
    rewards_learn = list()
    for j in range(len(checkpoints)):
        rewards_learn.append(checkpoints[j][1])
    max_number = np.max(rewards_learn)
    idx = np.where(rewards_learn == max_number)[0][0]
    checkpoint_dir = checkpoints[idx][0]
    print(f"best iteration {(idx+1)*10} rewards {max_number}")
    return checkpoint_dir

def find_latest_iteration(ck_path):
    ck_path = os.path.abspath(ck_path)
    # print(f"Received path: {ck_path}")
    analysis = tune.ExperimentAnalysis(ck_path)
    checkpoints = analysis.get_trial_checkpoints_paths(
        trial=analysis.get_best_trial("episode_reward_mean",mode="max"),
        metric="episode_reward_mean",
    )
    checkpoint_dir = checkpoints[-1][0]
    print(f"last iteration {(len(checkpoints))*10} rewards {checkpoints[-1][1]}")
    return checkpoint_dir


if __name__ == "__main__":

    args = get_args()
    args.randomize_loc = False if args.fix_loc else True
    os.makedirs(args.results_dir, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # Set up Ray.
    ray.init(log_to_driver=False, num_gpus=1)

    # Fetch experiment configurations
    config, env_config = get_config(args)
    # Start env
    ENV_CLASSES = {
        "MultiAgentSync_fullobs": MultiAgentSync_fullobs,
        "MultiAgentSing_fullobs": MultiAgentSing_fullobs,
        "MultiAgentSync_noobs": MultiAgentSync_noobs,
        "MultiAgentSing_noobs": MultiAgentSing_noobs,
        "MultiAgentSync_call": MultiAgentSync_call,
        "MultiAgentSing_call": MultiAgentSing_call,
        "MultiAgentSync_call2": MultiAgentSync_call2,
        "MultiAgentSing_call2": MultiAgentSing_call2,
        "MultiAgentSing_memory":MultiAgentSing_memory,
        "MultiAgentSync_memory":MultiAgentSync_memory
    }

    env_class = ENV_CLASSES.get(args.condition)
    env = env_class(config=env_config)

    # Find best agent
    checkpoint_dir = find_latest_iteration(args.checkpoint_dir)
    agent = PPOTrainer(config=config)
    agent.restore(checkpoint_dir)

    # run episodes
    # env.timestep_limit = 200
    obs = env.reset()
    state1 = [np.zeros(256, np.float32) for _ in range(2)]
    state2 = [np.zeros(256, np.float32) for _ in range(2)]

    while True:
        a1, state1, _ = agent.compute_single_action(
            observation=obs["agent1"], state=state1, policy_id="policy1"
        )
        #
        a2, state2, _ = agent.compute_single_action(
            observation=obs["agent2"], state=state2, policy_id="policy2"
        )
        obs, rewards, dones, _ = env.step({"agent1": a1, "agent2": a2})
        env.render_image()
        if dones["agent1"]:
            break
    save_video(
        env.image_list,
        output_path=f"{args.results_dir}/{os.path.basename(os.path.normpath(args.checkpoint_dir))}.mp4",
    )
