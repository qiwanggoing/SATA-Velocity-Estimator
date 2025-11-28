import os
import sys
# IMPORTANT: isaacgym must be imported before torch
import isaacgym 
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
import torch
from rsl_rl.modules import ActorCritic

def export_policy(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    
    # 1. Load the environment to get dimensions
    env, _ = task_registry.make_env(name=args.task, args=args)
    
    # 2. Initialize ActorCritic
    policy_cfg = class_to_dict(train_cfg.policy)
    if env.num_privileged_obs is not None:
        num_critic_obs = env.num_privileged_obs
    else:
        num_critic_obs = env.num_obs
    actor_critic = ActorCritic(env.num_obs, num_critic_obs, env.num_actions, **policy_cfg)
    actor_critic.to('cpu') # Export on CPU usually

    # 3. Load the trained weights
    # You need to manually set the path to your latest model_xxxx.pt here or via args
    # For this script, let's assume we load from the log directory provided in args
    load_run = args.load_run
    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
    
    # Find the latest run if not specified
    if load_run is None or load_run == "-1":
        # Filter out 'exported' and ensure we only look at directories
        runs = sorted([d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d)) and d != "exported"])
        if len(runs) == 0:
            print(f"No runs found in {log_root}")
            return
        load_run = runs[-1]
    
    model_path = os.path.join(log_root, load_run, f"model_{train_cfg.runner.max_iterations}.pt")
    print(f"Loading model from: {model_path}")
    
    loaded_dict = torch.load(model_path, map_location='cpu')
    actor_critic.load_state_dict(loaded_dict['model_state_dict'])
    actor_critic.eval()

    # 4. Trace the Actor (Policy) part
    # The actor takes 'observations' as input
    class PolicyExporter(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
        def forward(self, x):
            return self.actor(x)

    policy_exporter = PolicyExporter(actor_critic.actor)
    
    dummy_input = torch.randn(1, env.num_obs)
    traced_script_module = torch.jit.trace(policy_exporter, dummy_input)
    
    # 5. Save
    output_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, load_run, "policy_jit.pt")
    traced_script_module.save(output_path)
    print(f"Successfully exported JIT policy to: {output_path}")
    print(f"Don't forget to copy 'estimator_{train_cfg.runner.max_iterations}.pt' from the same folder!")

if __name__ == '__main__':
    args = get_args()
    export_policy(args)