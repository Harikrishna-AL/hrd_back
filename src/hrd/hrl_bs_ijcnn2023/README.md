# Supplementary Material for "Homeostatic Reinforcement Learning through Soft Behavior Switching with Internal Body State"

## Contents
* supplementary_materials: Supplementary materials of the paper
* envs: Implementation of benchmark environments
  * tiny_homeostasis: "LineHomeostatic-v0", "FieldHomeostatic-v0"
  * trp_env: "SmallLowGearAntTRP-v0"
  * thermal_regulation: "SmallLowGearAntTHR-v3"
* ppo.py: Reference PPO implementation
* ppo_mp.py: IBS and IMoE implementation

## Installation
```bash
conda create --name ibs_ijcnn2023 python=3.8
conda activate ibs_ijcnn2023
conda install tensorboard pytorch
pip install cffi==1.14.0 wandb

pip install -e envs/tiny_homeostasis 
pip install -e envs/trp_env
pip install -e envs/thermal_regulation 
```

## How to Run
```bash
# Line experiment
python ppo_mp.py --attention-policy --env-id LineHomeostatic-v0 --seed 0 --num-envs 10 --total-timesteps 5000000 --n-test-runs 10 --num-steps 5000 --max-test-steps 10000 --test-every-itr 1 --max-steps 10000

# Field experiment
python ppo_mp.py --attention-policy --env-id FieldHomeostatic-v0 --seed 0 --num-envs 10 --total-timesteps 1500000 --n-test-runs 10 --num-steps 5000 --max-test-steps 10000 --test-every-itr 1 --max-steps 10000

# Two-Resource Problem (with GPU)
python ppo_mp.py --attention-policy --seed 0 --cuda --gpu 0 --num-envs 10 --num-steps 30000 --total-timesteps 80000000

# Thermal Environment (with GPU)
python ppo_mp.py --attention-policy --env-id SmallLowGearAntTHR-v3 --seed 0 --cuda --gpu 0 --num-envs 10 --num-steps 30000 --total-timesteps 80000000
```

Tensorboard and optional W&B visualization gives performance plots
```bash
tensorboard --logdir runs
```

## Reference
[CleanRL](https://github.com/vwxyzjn/cleanrl): Reference implementation of PPO 


## Copyright
Copyright (c) 2023-Present, Naoto Yoshida @ Intelligent Systems and Informatics Laboratory, The University of Tokyo
All rights reserved.

## Citation
```
@INPROCEEDINGS{10191925,
  author={Yoshida, Naoto and Kanazawa, Hoshinori and Kuniyoshi, Yasuo},
  booktitle={2023 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Homeostatic Reinforcement Learning through Soft Behavior Switching with Internal Body State}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  keywords={Motor drives;Neural networks;Switches;Reinforcement learning;Benchmark testing;Autonomous agents;Behavioral sciences;Homeostatic Reinforcement Learning;Homeostasis;Deep Reinforcement Learning;Neural Architecture},
  doi={10.1109/IJCNN54540.2023.10191925}}
```
