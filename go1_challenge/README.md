# Go1 Challenge

## `isaaclab_tasks`
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


Test in the challenge env
```
go1_nav_env.py --level=1  --policy=...
```


