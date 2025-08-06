# Montreal Summer School 2025 - Go1 Locomotion Challenge

This repository contains code for the Go1 Locomotion Challenge, which involves training a Go1 robot to walk using reinforcement learning (RL). We will use:
- **Isaac Sim** [(https://github.com/isaac-sim/IsaacSim/)](https://github.com/isaac-sim/IsaacSim/) for simulation
- **Isaac Lab** [(https://github.com/isaac-sim/IsaacLab)](https://github.com/isaac-sim/IsaacLab) for the Go1 robot model
- **RSL RL library** [(https://github.com/leggedrobotics/rsl_rl/)](https://github.com/leggedrobotics/rsl_rl/) for the RL framework

## The Challenge

The core of the competition is to train a robust low-level walking policy in simulation, then perform sim-to-real transfer on an actual Unitree Go1 quadruped robot using the Isaac Lab framework.

Once that works, you'll have the opportunity to code a high-level, vision-based navigation controller to drive your robot autonomously in a simple obstacle race using monocular fisheye camera images. To make this task manageable, we will place AprilTags in the arena.

On Friday, the final challenge will evaluate your walking policy across three tiers of difficulty:

1. **No obstacles, joystick control** - Navigate a straight line to reach the goal as quickly as possible using joystick control.
2. **Obstacles, joystick control** - Same as Tier 1, but with obstacles like walls and uneven terrain in the arena.
3. **Obstacles, vision-based control** - Same as Tier 2, but without joystick control. Your robot navigates autonomously using images from a front-facing monocular fisheye camera. AprilTags with known positions are placed around the arena and near the goal, while tags with unknown positions mark large obstacles.

Good luck everyone!

## System Requirements

The main requirement is being able to run Isaac Sim and Isaac Lab. If you can successfully install and run these frameworks, your system should be compatible with this project.

**Recommended Operating System:** Ubuntu 22.04 LTS (this is what was used for testing and development)

Your system must meet Isaac Sim's [minimum hardware requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html#system-requirements), which include:
- NVIDIA RTX GPU (required for physics simulation)
- Sufficient RAM and storage
- Compatible NVIDIA drivers

Other Linux distributions and Windows may work but are not officially supported.

If your are not able to install the project on your own machine, the MILA will provide access to some computers.

## Installation


If you encounter any installation problems, please post your questions in the Discord channel.

### 1. Isaac Sim 4.5.0

**Isaac Sim 4.5.0** simulates the Go1 robot and environment physics.

1. Ensure your PC meets the [minimum system requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html#system-requirements).

2. Follow the Isaac Sim Documentation to install the **Workstation Setup**: 
   [Workstation Installation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html#workstation-setup).

3. Test that the application starts:
   ```bash
   ./isaacsim/isaac-sim.sh
   ```

### 2. Isaac Lab

**Isaac Lab** is a modular robot learning framework built on Isaac Sim. It's used to create RL environments and control the simulation.

1. Follow the instructions to install Isaac Lab from binaries: [Isaac Lab Binaries Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html).

> [!WARNING] 
> Make sure you set up the Conda environment ([Setting up the conda environment](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#setting-up-the-conda-environment-optional)). A virtual environment is required to install the other projects.

2. Verify your installation ([Verifying the Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#verifying-the-isaac-lab-installation)):

   ```bash
   # With your conda environment activated, from the IsaacLab root folder
   python scripts/tutorials/00_sim/create_empty.py
   ```

> [!NOTE] 
> Isaac Lab also supports installation from `pip`. However, this has NOT been tested with this project. You can try it, but we won't be able to provide support if you follow this approach.

> [!WARNING]
> The installation script doesn't work when called from `zsh` (see [\[Bug Report\] Conda environment not being setup correctly for ZSH · Issue #703 · isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab/issues/703)).
> We recommend running with `bash`.
> 
> Alternatively, if you *really* want to use `zsh`:
> 1. Modify `<isaac_lab_path>/isaaclab.sh` line 19 to:
>    ```
>    export ISAACLAB_PATH="$( cd "$( dirname "$0" )" &> /dev/null && pwd )"
>    ```
> 2. Modify how `SCRIPT_DIR` is computed in `<isaac_lab_path>/_isaac_sim/setup_conda_env.sh` and `<isaac_lab_path>/_isaac_sim/setup_python_env.sh` to:
>    ```
>    # Determine the script directory
>    if [ -n "$ZSH_VERSION" ]; then
>      SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
>    else
>      SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
>    fi
>    ```

### 3. Go1 Challenge

Install the Go1 Challenge repository:

1. Clone the repository:
   ```bash
   git clone https://github.com/modanesh/MRSS2025-go1-challenge.git
   ```

2. Install as a Python package:
   ```bash
   # Activate your conda environment created for IsaacLab (if not already activated)
   conda activate isaaclab

   cd MRSS2025-go1-challenge

   # Install the package
   pip install -e .
   ``` 

3. Test your installation:
   ```bash
   python scripts/00-verify_installation.py
   ```

### 4. VS Code Setup (Optional)
TODO
- Settings
- Launch configuration

## Challenge 1 - Learn to Walk

### 1. Training Policy

If you've followed all the steps correctly, you should have a Python environment (Conda) with Isaac Lab installed along with this project.

Navigate to the project folder and run this command to train a locomotion policy for the Go1 robot:

```bash
python scripts/01-train.py --config=config/training_params.yaml
```

The `--config` flag specifies the configuration file for training parameters. See the `config` folder for more details.

During training, you can monitor progress in the WandB dashboard. Make sure you have WandB installed and configured:

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

Checkpoints will be saved under `logs/rsl_rl/<run_name>/<timestamp>`.

> [!INFO] 
> **Continue Training**
> You can continue training by loading either a run name or a path to a checkpoint:
> - To load an RSL-RL checkpoint: `--load_run 2025-07-24_13-11-02`. This loads the best policy from the run. The run must be in the RSL log directory.
> - To load a specific policy: `--checkpoint path_to_policy/policy_to_load.pt`

> [!INFO]
> **Training Configuration**
> Here are some tips to improve your policy:
> - Modify the terrain settings
> - Adjust the reward coefficients

### 2. Playing Policy

To test the trained policy, run the following command:

```bash
python scripts/01-play.py --load_run=RUN_NAME
```

Available arguments:
- `--load_run`: Specifies the run to load (folder name where the trained policy is saved)
- `--num_envs`: Number of environments to run in parallel
- `--headless`: Runs the simulation without a GUI

You have two options to load checkpoints:
- `--load_run 2025-08-05_15-16-27`: Runs the latest policy from that run
- `--checkpoint path/to/policy/my_policy.pt`: Loads a specific policy file

> [!NOTE]
> This script also exports the policy as a `jit` file, which is necessary for loading on the robot.

> [!NOTE]
> The play version uses fewer environments (defaults to 50) and a smaller terrain to reduce memory usage. If it's still too demanding for your laptop, reduce it with `--num_envs=10`.

The script exports your policy to a `jit` file at `logs/rsl_rl/<run_name>/<timestamp>/exported/policy.pt`.

### 3. Deploy

After verifying your policy works well in simulation, you can test it on the real robot! Send your exported policy (e.g., `policy.pt`) to your robot supervisor.

## Challenge 2 - Navigate Obstacles
*This part is in progress*

Train a policy that performs well on uneven terrain and across obstacles.

You can test your policy's performance in simulation by changing the terrain level. The terrain configuration is defined in `go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg.ROUGH_TERRAINS_CFG`. Try changing the proportions of different difficulties in the configuration file (under `env.terrain.terrain_generator.sub_terrains`).

You can also test your policy in the arena with:
```bash
python scripts/03-go1_arena.py --teleop --level 1 --policy logs/rsl_rl/go1_locomotion/2025-08-05_15-16-27_go1_locomotion/exported/policy.pt
```

Robot controls:
- **Arrow keys**: Linear velocity
- **Z & X**: Yaw rotation
- **R**: Reset
- **ESC**: Close simulation

There are three levels of increasing difficulty:
- 1: No obstacles, flat ground
- 2: Obstacles, flat ground
- 3: Obstacles, rough ground

The level can be specified via the `--level` arg.  



## Challenge 3 - Vision-Based Navigation

Coming soon...

## Code Overview

### Project Structure
```
├── config
├── go1_challenge
│   ├── arena_assets
│   ├── isaaclab_tasks
│   ├── navigation
│   ├── utils
│   └── scripts
└── scripts
    ├── camera
    ├── rsl_rl
    └── skrl
```

- `config`: YAML configuration files for environment and training parameters
- `go1_challenge`: Implementation of the Gym environment
- `scripts`: Executable scripts for training, playing, and testing

### Go1_challenge - Python Package

#### `arena_assets`
Assets and scripts for generating the competition arena.

#### `isaaclab_tasks`
Isaac Lab environments implemented as manager-based environments. These are modifications of the [Isaac-Velocity-Rough-Unitree-Go1-v0](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py) environment.

- `go1_locomotion_env_cfg.py`: Environment implementation using configuration files
- `go1_challenge_env_cfg.py`: Challenge environment implementation
- `mdp`: Implementation of MDP functions including reward terms, reset conditions, and terrain difficulty
- `agents`: Configuration for the [RSL-RL](https://github.com/leggedrobotics/rsl_rl) agent
