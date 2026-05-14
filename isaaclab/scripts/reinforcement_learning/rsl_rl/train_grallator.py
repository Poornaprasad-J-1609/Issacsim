import sys
import runpy

# Put your local Grallator package first
sys.path.insert(0, "/workspace/isaaclab/grallator_quadruped_isaac")

# Register local Grallator Gym tasks
import grallator  # noqa: F401

# Make train.py local imports like cli_args.py work
sys.path.insert(0, "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl")

# Launch official Isaac Lab RSL-RL train.py
runpy.run_path(
    "/workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py",
    run_name="__main__",
)
