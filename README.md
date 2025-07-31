# MRSS 2025 --- Go1 Challenge
IsaacSim project for the MRSS 2025 Go1 Challenge.

Project Structure
```
├── config
├── go1_challenge
│   ├── isaaclab_tasks
│   ├── isaac_sim
│   ├── ros_pkg
│   └── scripts
├── policies
└── scripts
    ├── camera
    ├── rsl_rl
    └── skrl
```

- `config`: `.yaml` configuration files for the environment and training parameters
- `go1_challenge`: Implementation of the `gym` environment
- `scripts`:


## Go1_challenge
## `isaac_sim`
Assets and scripts for the generation of the competition arena.

This is where `Go1ChallengeSceneCfg`, the competition environment is defined. 

## `isaaclab_tasks`
Implementation of the IsaacLab environments. 

The Isaac Lab environment are implemented here as manager-based environments. They are modification of the [Isaac-Velocity-Rough-Unitree-Go1-v0](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py) environment. 


`go1_locomotion_env_cfg.py`
Implementation of the environment using configuration files. 

`mdp`
Implementation of the functions required in the mdp like computing the reward terms, reset conditions and terrain 
difficulty.

`agents`
Configuration of the *[rsl-rl](https://github.com/leggedrobotics/rsl_rl)* agent.


## Usage
Training
```
train.py --config=...
```

Play the policies
You can test how the policy is performing in the environment
```
play.py --config=
```
> [!Note]
> The play version of the task has less environments and smaller terrain to reduce memory usage.

Args:
- To load a rsl-rl checkpoint: `--load_run 2025-07-24_13-11-02`. This will load the best policy in the run. 
  The run needs to be in the rsl log directory
- To load a specific policy: `--checkpoint path_to_policy/policy_to_load.pt`


Test in the challenge env
```
go1_nav_env.py --level=1  --policy=...
```


## Tags
Generated with https://chev.me/arucogen/

## Installation
Dependencies (to add in the pyproject)
- pyapriltags
<!-- - opencv-python -->


1. Install Isaac Sim 4.5.0 from binaries
   https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html#workstation-setup

2. Install Isaac Lab
3. 