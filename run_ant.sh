#!/bin/bash
SEEDS="0 1 2 3 4"
for SEED in $SEEDS
do
	python baselines/gail/run_mujoco.py --env_id Ant-v1 --seed $SEED --num_epochs 400 --timesteps_per_batch 5000 --eval_interval 4 --expert_path dataset/ant.npz
done