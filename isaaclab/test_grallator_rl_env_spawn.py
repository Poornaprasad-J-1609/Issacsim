from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True, livestream=1)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


TASK = "Isaac-Velocity-Flat-Grallator-v0"

env_cfg = parse_env_cfg(
    TASK,
    device="cuda:0",
    num_envs=4,
)

env = gym.make(TASK, cfg=env_cfg)

print("\n====================================")
print("GRALLATOR RL ENV SPAWN TEST")
print("====================================")
print("Task:", TASK)
print("Num envs:", env.unwrapped.num_envs)
print("Action dim:", env.unwrapped.action_manager.total_action_dim)

obs, info = env.reset()

action_dim = env.unwrapped.action_manager.total_action_dim
actions = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)

for i in range(300):
    obs, reward, terminated, truncated, info = env.step(actions)

    if i % 50 == 0:
        root_pos = env.unwrapped.scene["robot"].data.root_pos_w
        print(f"step={i:04d} root_z={root_pos[:, 2].detach().cpu().numpy()}")

print("\nRL env spawn test finished. Keeping viewer open.")

while simulation_app.is_running():
    obs, reward, terminated, truncated, info = env.step(actions)

env.close()
simulation_app.close()
