# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import carb
import omni.appwindow

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # ---------------------------------------------------------------------
    # Grallator keyboard command override
    # W/S : forward/backward vx
    # A/D : left/right vy
    # Q/E : yaw left/right wz
    # SPACE : stop
    # ---------------------------------------------------------------------
    pressed_keys = set()

    max_vx = 1.4
    max_vy = 0.8
    max_wz = 0.7

    def keyboard_event_callback(event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            pressed_keys.add(event.input)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            pressed_keys.discard(event.input)
        return True

    app_window = omni.appwindow.get_default_app_window()
    keyboard = app_window.get_keyboard()
    input_interface = carb.input.acquire_input_interface()
    keyboard_sub = input_interface.subscribe_to_keyboard_events(keyboard, keyboard_event_callback)

    def apply_keyboard_command():
        vx = 0.0
        vy = 0.0
        wz = 0.0

        if carb.input.KeyboardInput.W in pressed_keys:
            vx += max_vx
        if carb.input.KeyboardInput.S in pressed_keys:
            vx -= max_vx

        # Isaac convention: +Y is left, -Y is right.
        if carb.input.KeyboardInput.A in pressed_keys:
            vy += max_vy
        if carb.input.KeyboardInput.D in pressed_keys:
            vy -= max_vy

        if carb.input.KeyboardInput.Q in pressed_keys:
            wz += max_wz
        if carb.input.KeyboardInput.E in pressed_keys:
            wz -= max_wz

        if carb.input.KeyboardInput.SPACE in pressed_keys:
            vx = 0.0
            vy = 0.0
            wz = 0.0

        command_manager = env.unwrapped.command_manager
        command_names = list(command_manager._terms.keys())

        if "base_velocity" not in command_manager._terms:
            raise RuntimeError(f"Could not find command term 'base_velocity'. Available command terms: {command_names}")

        cmd_term = command_manager._terms["base_velocity"]
        cmd_term.command[:, 0] = vx
        cmd_term.command[:, 1] = vy
        cmd_term.command[:, 2] = wz

    print("[GRALLATOR KEYBOARD CONTROL]")
    print("  W/S   : forward/backward")
    print("  A/D   : left/right")
    print("  W+D   : diagonal forward-right")
    print("  W+A   : diagonal forward-left")
    print("  Q/E   : yaw left/right")
    print("  SPACE : stop")
    print("Click inside the livestream/viewer first, then press keys.")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            apply_keyboard_command()
            # refresh obs so policy sees latest keyboard command
            obs = env.get_observations()
            # agent stepping
            actions = policy(obs)

            # =========================================================
            # SIM2REAL_PER_JOINT_LOGGER
            # Full policy action, no artificial action clipping.
            # Logs each joint target range for real robot limits.
            # =========================================================
            if "sim2real_step" not in locals():
                sim2real_step = 0

            JOINT_ORDER_LOG = [
                "FL_hip", "FR_hip", "RL_hip", "RR_hip",
                "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
                "FL_calf", "FR_calf", "RL_calf", "RR_calf",
            ]

            Q_DEFAULT_LOG = torch.tensor(
                [0.0, 0.0, 0.0, 0.0,
                 0.1, -0.1, 0.1, -0.1,
                 0.0, 0.0, 0.0, 0.0],
                device=actions.device,
                dtype=actions.dtype,
            )

            ACTION_SCALE_LOG = 0.12
            q_target_log = Q_DEFAULT_LOG + ACTION_SCALE_LOG * actions

            if sim2real_step % 100 == 0:
                a = actions.detach().cpu()
                q = q_target_log.detach().cpu()
                print("\n[SIM2REAL_LOG] step=", sim2real_step)
                print("action global: min=%.3f max=%.3f mean_abs=%.3f" %
                      (a.min().item(), a.max().item(), a.abs().mean().item()))
                for ji, jn in enumerate(JOINT_ORDER_LOG):
                    print("%02d %-10s action[min,max]=[% .3f,% .3f] q_target[min,max]=[% .3f,% .3f]" %
                          (ji, jn, a[:, ji].min().item(), a[:, ji].max().item(),
                           q[:, ji].min().item(), q[:, ji].max().item()))

            sim2real_step += 1


            # =========================================================
            # SIM-TO-REAL ACTION LOGGER
            # Full policy action, no artificial action clipping.
            # Logs action range every 100 steps.
            # =========================================================
            if timestep % 100 == 0:
                a = actions.detach().cpu()
                print(
                    f"[ACTION_LOG] step={timestep} "
                    f"min={a.min().item(): .3f} "
                    f"max={a.max().item(): .3f} "
                    f"mean_abs={a.abs().mean().item(): .3f}"
                )

            
            # =========================================================
            # SIM-TO-REAL SAFETY ACTION CLIP
            # Start real transfer with 0.50; later test 0.75 and 1.00
            # =========================================================
            ACTION_CLIP = 1.00
            # # # actions = actions.clamp(-ACTION_CLIP, ACTION_CLIP)  # disabled: full walking policy  # disabled for walking  # disabled: use full trained policy action
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
