# Sim2Real Tutorial

> [!NOTE] 
> Instructions on how to reproduce the sim2real demonstration scenarios.

## Overview

This showcases three different policies trained with varying levels of domain randomization and terrain complexity:

1. **No Domain Randomization (Baseline)**: Policy trained on flat terrain without randomization
2. **Domain Randomization on Flat Terrain**: Policy with mass randomization on flat terrain  
3. **Domain Randomization on Rough Terrain**: Policy trained on varied terrain with mass randomization

## Training Policies

All policies are trained using the parameter file `config/sim2real_params.yaml`:

```bash
python scripts/01-train.py --config=config/sim2real_params.yaml
```

### Policy Specifications

We trained three different policies. They are located in `policies/sim2real_tutorial/`

#### `policy_flat_no_DR` (Baseline)
- **Training Duration**: 1000 iterations
- **Terrain**: Flat terrain only
- **Domain Randomization**: None (`mass_distribution_params: 0, 0`)

#### `policy_flat_DR` (Domain Randomization)
- **Training Duration**: 1000 iterations  
- **Terrain**: Flat terrain only
- **Domain Randomization**: Mass variation (`mass_distribution_params: (-1, 3)`)

#### `policy_rough_DR` (Full Randomization)
- **Training Duration**: 1500 iterations
- **Terrain**: Mixed terrain types (flat, rough, stairs, ...)
- **Domain Randomization**: Mass variation (`mass_distribution_params: (-1, 3)`) 

## Validation Environment

Use the validation environment `MRSS-Velocity-Go1-Sim2Real-v0` to test policy robustness:

```bash
python scripts/02-play.py --task "MRSS-Velocity-Go1-Sim2Real-v0" --checkpoint policies/sim2real_tutorial/policy_flat_DR/model.pt
```

### Modifying Test Conditions

**Added Mass**: 
- Training: Modify `mass_distribution_params` at line 51 in `configs/sim2real_params.yaml`
- Validation: Modify the `added_mass` parameter at line 122 in `go1_sim2real_env_cfg.py` 

**Terrain**:
- Training: Modify the proportions of the subterrains in `configs/sim2real_params.yaml` (lines 25-45)

**Recording Videos**: Capture demonstrations using:
```bash
python scripts/02-play.py --task "MRSS-Velocity-Go1-Sim2Real-v0" --checkpoint <policy_path> --video --video_length <length>
```
Where `<length>` is the number of steps (100 steps = 2 seconds).

## Demonstration Scenarios
*Videos of the results are located at `docs/sim2real/`*

### Scenario 1: `01_flat_no_dr_nominal_mass`
- **Policy**: `policy_flat_no_DR`
- **Mass**: Nominal (no added mass)

### Scenario 2: `02_flat_no_dr_added_mass`
- **Policy**: `policy_flat_no_DR` 
- **Mass**: +3kg added mass

### Scenario 3: `03_flat_dr_nominal_mass`
- **Policy**: `policy_flat_DR`
- **Mass**: Nominal (no added mass)

### Scenario 4: `04_flat_dr_added_mass`  
- **Policy**: `policy_flat_DR`
- **Mass**: +3kg added mass

### Scenario 5: `05_rough_dr_added_mass`
- **Policy**: `policy_rough_DR` 
- **Mass**: +3kg added mass
