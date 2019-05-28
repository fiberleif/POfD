#!/bin/bash
SEEDS="0 1 2 3 4"
for SEED in $SEEDS
do
	python baselines/gail/run_mujoco.py --env_id Hopper-v1 --seed $SEED --num_epochs 1000
done