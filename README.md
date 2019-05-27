# Policy Optimization with demonstrations (POfD)

This repository is a reimplementation of [Policy Optimization with demonstrations](http://proceedings.mlr.press/v80/kang18a.html) (ICML 2018).

## Dependencies
This code is highly based on [OpenAI baselines gail](https://github.com/openai/baselines/tree/master/baselines/gail).

## Training

To run `POfD` on delayed Mujoco tasks:
```angular2html
python -m baselines.gail.run_mujoco --env Hopper-v1 --reward-freq 10
```
