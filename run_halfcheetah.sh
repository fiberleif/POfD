#!/bin/bash
SEEDS="0 1 2 3 4"
for SEED in $SEEDS
do
	python baselines/gail/run_mujoco.py --env_id HalfCheetah-v1 --seed $SEED --num_epochs 1000 --reward_coeff 0.0 --expert_path dataset/half_cheetah.npz
done