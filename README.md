# Policy Optimization with Demonstrations (POfD)

This repository is a reimplementation of [Policy Optimization with Demonstrations](http://proceedings.mlr.press/v80/kang18a.html) (ICML 2018).

## Dependencies
This code is highly based on [OpenAI baselines gail](https://github.com/openai/baselines/tree/master/baselines/gail).

## Training

To run `POfD` on delayed Mujoco tasks:
```angular2html
python baselines/gail/run_mujoco.py --env_id Hopper-v1 --reward-freq 10 --num_epochs 1000
```
