# Montreal Summer School 2025 - Go1 Locomotion Challenge

This repository contains the code for the Go1 Locomotion Challenge, involving training a Go1 robot to walk using RL. To do so, we will use Isaac Sim [(https://github.com/isaac-sim/IsaacSim/)](https://github.com/isaac-sim/IsaacSim/) for simulation, Isaac Lab [(https://github.com/isaac-sim/IsaacLab)](https://github.com/isaac-sim/IsaacLab) for the Go1 robot model, and the RSL RL library [(https://github.com/leggedrobotics/rsl_rl/)](https://github.com/leggedrobotics/rsl_rl/) for RL framework.


## The Challenge
The core of the competition is to train a robust (low-level) walking policy in simulation, and then perform sim-to-real on an actual Unitree Go1 quadruped robot. You will achieve this thanks to the Isaac Lab framework. 

Once that works, you will have the opportunity to code a high-level, vision-based navigation controller driving your robot in a simple obstacle race, autonomously, from monocular fisheye camera images. To make this task manageable, we will place AprilTags in the arena.

On Friday, the final challenge will evaluate your walking policy on a series of obstacle races. The challenge has 3 tiers, from easiest to most difficult:
1. **No obstacle, joystick control** - Your walking policy is evaluated on a simple straight line, reaching the goal as fast as possible. You control the robot via a joystick.
2. **Obstacles, joystick control** - Same setting as Tier 1, but obstacles are placed in the arena. Obstacles can be things like walls and uneven floor.
3. **Obstacles, vision-based control** - Same setting as Tier 2, but you do not have a joystick. Instead, your robot navigates autonomously, from images captured by a front-facing monocular fisheye camera. To help you navigate, tags with known positions are placed around the arena and near the goal. Tags with unknown positions are also placed on large obstacles.

Good luck everyone!

## Installation
If you have any problems with the installation, post your questions in the discord channel. 

### 1. Isaac Sim 4.5.0
**Isaac Sim 4.5.0** is used to simulate the Go1 and environments physics. 

1. Make sure your PC hat at least has the [minimum specs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html#system-requirements).


2. Follow the instructions from the Isaac Sim Documentation to install the **Workstation Setup** of the software: 
[Workstation Installation](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html#workstation-setup).

3. Test the app starts
```bash
./isaacsim/isaac-sim.sh
```

### 2. Isaac Lab
**Isaac Lab** is a modular robot learning framework built on *Isaac Sim*. It's used to create the RL environments and
control the simulation. 

1. Follow the instructions to install Isaac Lab from binaries: [IsaacLab Binaries Installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html).

> [!warning] 
> Make sure you setup the Conda environment ([Setting up the conda environment](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#setting-up-the-conda-environment-optional)). A virtual environment
> is needed to install the other projects.



2. Verify your installation ([Verifying the Isaac Lab installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html#verifying-the-isaac-lab-installation)).

```bash
# With your conda env activated, from IsaacLab root folder
python scripts/tutorials/00_sim/create_empty.py
```

> [!note] 
> Isaac Lab also support installation from `pip`. However, it has NOT been tested with this project. You can try it, but
> we won't be able to support you if you follow this approach. 

> [!warning] Using zsh?
> The installation script doesn't work when called from `zsh` (see [\[Bug Report\] Conda environment not being setup correctly for ZSH · Issue #703 · isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab/issues/703)).
> Is is recommended you run using `bash`.
> Alternatively, if you *really* want to use `zsh`:
> 1. Modify `<isaac_lab_path>/isaaclab.sh` line 19 to:
>  ```
>  export ISAACLAB_PATH="$( cd "$( dirname "$0" )" &> /dev/null && pwd )"
>  ```
> 2. Modify how `SCRIPT_DIR` is computed in `<isaac_lab_path>/_isaac_sim/setup_conda_env.sh` and `<isaac_lab_path>/_isaac_sim/setup_python_env.sh` to:
> ```
> # Determine the script directory
> if [ -n "$ZSH_VERSION" ]; then
 >   SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"
> else
 >   SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
> fi
> ```
>  


### 3. Go1 Challenge
Next step is to install the Go1 Challenge repo.

1. Clone the repository:
```bash
git clone https://github.com/modanesh/MRSS2025-go1-challenge.git
```

2. Install as a python package
```bash
# If not already activated, activate your conda env created for IsaacLab 
conda activate isaaclab

cd MRSS2025-go1-challenge

# Install the package
pip install -e .
``` 

3. Test your installation
```bash
python scripts/00-verify_installation.py
```

### 4. Vs Code Setup (optional)
TODO
- settings
- launch


## Challenge 1 - Learn to walk
### 1. Training Policy
If all the steps are followed correctly, you have a python environment (Conda) with Isaac Lab installed along this project.

Then, go to the project's folder, and run this command to train a locomotion policy for the Go1 robot:

```bash
python scripts/01-train.py --config=config/training_params.yaml
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

This will save checkpoints under `logs/rsl_rl/<run_name>/<timestamps>`.


> [!tip] Continue training
> You can continue loading by either passing a run name or a path to a checkpoint:
> - To load a rsl-rl checkpoint: `--load_run 2025-07-24_13-11-02`. This will load the best policy in the run. 
  The run needs to be in the rsl log directory
> - To load a specific policy: `--checkpoint path_to_policy/policy_to_load.pt`

> [!tip] Training Config
> Here are some pointers to improve your policy: 
> - Change the terrain
> - Play with the reward coefficients



### 2. Playing Policy

To play the trained policy, you can run the following command:

```bash
python scripts/01-play.py --load_run=RUN_NAME
```
Other args:
- `--load_run` specifies the run to load, which should be the name of the folder where the trained policy is saved. 
- `--num_envs` specifies the number of environments to run in parallel.
- `--headless` runs the simulation without a GUI.

You have two different options to load checkpoints.
- `--load_run 2025-08-05_15-16-27`: This runs the latest policy in that run
- `--checkpoint path/to/policy/my_policy.pt`: Loads the policy from a file

> [!note]
> This script also exports the policy as `jit`. 


For the rest of the parameters, you can refer to the `play.py` script.

> [!Note]
> The play version of the task has less environments (defaults to 50) and smaller terrain to reduce memory usage. If it's still too much for your laptop, reduce it with `--num_envs=10`.

This script also exports your policy to a `jit` file. This is necessary to load it on the robot. The policy is exported to 
`logs/rsl_rl/<run_name>/<timestamps>/exported/policy.pt`.

### 3. Deploy
After making sure your policy looks good in simulation, you can test it on the real robot! Send your exported policy (eg: `policy.pt`) to your robot supervisor. 


## Challenge 2 - Robust Walking
The next step is to train a policy that's also able to perform well on uneven terrain and across obstacles. 

You can test how your policy is doing in simulation by changing the terrain level. The terrain configuration is defined
in `go1_challenge.isaaclab_tasks.go1_locomotion.go1_locomotion_env_cfg.ROUGH_TERRAINS_CFG. You can try changing the 
proportions of the different difficulties in the configuration file (under `env.terrain.terrain_generator.sub_terrains`)


You can also test your policy in the arena with:
```bash
python scripts/03-go1_arena.py --teleop --level 1 --policy logs/rsl_rl/go1_locomotion/2025-08-05_15-16-27_go1_locomotion/exported/policy.pt
```

You can control the robot with:
- Arrows: Linear velocity
- `Z & X`: Yaw.
- `R`: Reset.
- `ESC`: Close the sim. 

There are three levels of increasing difficulty:
- 1: No obstacles, flat ground
- 2: Obstacles, flat ground
- 3: Obstacles, rough ground

The level can be specified via the `--level` arg.  


## Challenge 3 - Walking alone
...


## Code Overview
Project Structure
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

- `config`: `.yaml` configuration files for the environment and training parameters
- `go1_challenge`: Implementation of the `gym` environment
- `scripts`


### Go1_challenge - Python Package
## `arena_assets`
Assets and scripts for the generation of the competition arena.


## `isaaclab_tasks`
The Isaac Lab environment are implemented here as manager-based environments. They are modifications of the [Isaac-Velocity-Rough-Unitree-Go1-v0](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py) environment. 


- `go1_locomotion_env_cfg.py`: Implementation of the environment using configuration files. 

- `go1_challenge_env_cfg.py`: Implementation of the challenge environment. 

- `mdp`: Implementation of the functions required in the mdp like computing the reward terms, reset conditions and terrain 
difficulty.

- `agents`: Configuration of the *[rsl-rl](https://github.com/leggedrobotics/rsl_rl)* agent.


