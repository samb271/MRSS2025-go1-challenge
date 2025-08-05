# Montreal Summer School 2025 - Go1 Locomotion Challenge

This repository contains the code for the Go1 Locomotion Challenge, involving training a Go1 robot to walk using RL. To do so, we will use Isaac Sim [(https://github.com/isaac-sim/IsaacSim/)](https://github.com/isaac-sim/IsaacSim/) for simulation, Isaac Lab [(https://github.com/isaac-sim/IsaacLab)](https://github.com/isaac-sim/IsaacLab) for the Go1 robot model, and the RSL RL framework [(https://github.com/leggedrobotics/rsl_rl/)](https://github.com/leggedrobotics/rsl_rl/) for RL framework.


## Installation

First step is to clone this repository:

```bash
git clone https://github.com/modanesh/MRSS2025-go1-challenge.git
```

Then, given your system configuration, follow the these instructions here to install the Isaac ecosystem:
[Isaac Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)


### Training Policy

If all the steps are followed correctly, you should have the Isaac Lab installed in the `MRSS2025-go1-challenge` directory. Then, go to the project's folder, and run this command to train a locomotion policy for the Go1 robot:

```bash
./IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py --config=config/training_params_mrss.yaml
```

`--config` specifies the configuration file for the training parameters. Look at the `config` folder for more details.

During the training, you can monitor the training progress in the WandB dashboard. Make sure you have WandB installed and configured. You can install it using pip:

```bash
pip install wandb
```

Here are some plots you should expect to see in the WandB dashboard:

<details>
  <summary>Click to view the WandB Dashboard plot</summary>

  <p>
    <img src="./config/wandb_plots.png" alt="WandB Dashboard" />
  </p>
</details>

### Playing Policy

To play the trained policy, you can run the following command:

```bash
./IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py --task=MRSS-Velocity-Go1-Play-v0 --load_run=RUN_NAME --num_envs=128 --headless
```

`--task` specifies the task to play, which in this case is `MRSS-Velocity-Go1-Play-v0`. 

`--load_run` specifies the run to load, which should be the name of the folder where the trained policy is saved. 

`--num_envs` specifies the number of environments to run in parallel.

`--headless` runs the simulation without a GUI, which is useful for running on servers or headless machines.


For the rest of the parameters, you can refer to the `play.py` script.
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