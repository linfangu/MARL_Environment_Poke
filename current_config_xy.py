from MultiAgentSync_fullobs_samefield_randnose_xycoords import (
    MultiAgentSync_fullobs,
    MultiAgentSing_fullobs,
)
from ray.rllib.models import ModelCatalog
from ray.rllib.agents.ppo import PPOTorchPolicy
from Customcallback import CustomCallbacks
from attention_rnn import AnotherTorchRNNModel
import torch


def get_config(args=None):
    args = args or type("Args", (), {})()
    env_config = {
        "height": 8,
        "width": 8,
        "sync_limit": getattr(args, "coop_window",2),
        "randomize": getattr(args, "randomize_loc",True),
        "randomize_miss": getattr(args, "randomize_miss",False),
        "miss_reward": getattr(args, "miss_reward", -0.5),
        "Water1": [7, 2],
        "Water2": [7, 6],
        "pretrain":getattr(args, "pretrain", False)
    }
    ENV_CLASSES = {
        "MultiAgentSync_fullobs": MultiAgentSync_fullobs,
        "MultiAgentSing_fullobs": MultiAgentSing_fullobs,
    }

    env_class = ENV_CLASSES.get(args.condition)
    if env_class:
        env = env_class(config=env_config)
    else:
        raise ValueError(f"Environment '{args.condition}' not found.")

    ModelCatalog.register_custom_model("rnn2", AnotherTorchRNNModel)

    policies = {
        "policy1": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
        "policy2": (PPOTorchPolicy, env.observation_space, env.action_space, {}),
    }

    # 2) Defines an agent->policy mapping function.
    def policy_mapping_fn(agent_id: str) -> str:
        # Make sure agent ID is valid.
        assert agent_id in ["agent1", "agent2"], f"ERROR: invalid agent ID {agent_id}!"
        ### Modify Code here ####
        id = agent_id[-1]
        return f"policy{id}"

    config = {
        "env": env_class,
        "env_config": env_config,
        "num_workers": 0,
        "exploration_config": {
            "type": "Curiosity",  # <- Use the Curiosity module for exploring.
            "eta": 1.5,  # Weight for intrinsic rewards before being added to extrinsic ones.
            "lr": 0.001,  # Learning rate of the curiosity (ICM) module.
            "feature_dim": 64,  # Dimensionality of the generated feature vectors.
            # Setup of the feature net (used to encode observations into feature (latent) vectors).
            "feature_net_config": {
                "use_lstm": False,
            },
            "inverse_net_hiddens": [256],  # Hidden layers of the "inverse" model.
            "inverse_net_activation": "relu",  # Activation of the "inverse" model.
            "forward_net_hiddens": [256],  # Hidden layers of the "forward" model.
            "forward_net_activation": "relu",  # Activation of the "forward" model.
            "beta": 0.2,  # Weight for the "forward" loss (beta) over the "inverse" loss (1.0 - beta).
            "sub_exploration": {
                "type": "StochasticSampling",
            },
        },
        # !PyTorch users!
        "framework": "torch",  # If users have chosen to install torch instead of tf.
        "create_env_on_driver": True,
    }

    max_seq_len = 10

    # 3) RNN config.
    config.update(
        {
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
            },
            "model": {
                "custom_model": "rnn2",
                "max_seq_len": max_seq_len,
                "custom_model_config": {
                    "rnn_hidden_size": 256,
                    "l2_lambda": getattr(args,"l2_curr",0.1),
                    "l2_lambda_inp": getattr(args,"l2_inp",0),
                    "noise_std": 0,
                    "device": torch.device("cuda:0"),
                },
            },
            "num_workers": 0,
            "num_gpus": getattr(args,"num_gpus",0.3),
            "callbacks": CustomCallbacks,
        }
    )

    print()
    print(f"agent1 is now mapped to {policy_mapping_fn('agent1')}")
    print(f"agent2 is now mapped to {policy_mapping_fn('agent2')}")
    return config, env_config
