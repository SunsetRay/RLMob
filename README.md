# RLMob
Code for the paper "RLMob: Deep Reinforcement Learning for Successive Mobility Prediction" 

- run run.py to launch the framework (global parameters in run.py), recommend to use 
```
nohup python -u run.py >log/F_ppo.log 2>&1 &
```
- run pretrain/main.py to pretrain (parameters are at the top of train.py), and pretrain/parameters.py for adjusting hyperparameters for pretraining
