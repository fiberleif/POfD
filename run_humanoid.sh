#!/bin/bash
SEEDS="0 1 2 3 4"
for SEED in $SEEDS
do
	python baselines/gail/run_mujoco.py --env_id Humanoid-v1 --seed $SEED --num_epochs 800 --timesteps_per_batch 5000 --eval_interval 4 --expert_path dataset/humanoid.npz
done