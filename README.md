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
    ![WandB Dashboard](./config/wandb_plots.png)
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