#!/bin/bash
REWARD_COEFFS="0.0 0.2 0.4 0.6 0.8 1.0"
SEEDS="0 1 2 3 4"
for REWARD_COEFF in $REWARD_COEFFS
do
    for SEED in $SEEDS
    do
	    python baselines/gail/run_mujoco.py --env_id Humanoid-v1 --seed $SEED --num_epochs 800 --reward_coeff $REWARD_COEFF --timesteps_per_batch 5000 --eval_interval 4 --expert_path dataset/humanoid.npz
    done
done