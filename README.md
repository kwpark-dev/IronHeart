# IronHeart

In this project, several representatives of modern RL approaches, especially deep learning-based, will be addressed. The learning capability is examined in Mujoco environments which contain continuous action and state space: InvertedPendulum-v5, Reacher-v5, and HalfCheetah-v5. The objectives are as follows.

1. Brief taxonomy
2. The strategies and the implications
3. Implementation from scratch

## Deep Deterministic Policy Gradient (DDPG)

| Type        | Training Policy | Execution Policy | Sampling | Remark |
|-------------|-----------------|-------------|---------------|-------|
| Off-policy  | Deterministic  | Deterministic | Implicit | Target Network|

The essence of DDPG is target networks which allows rather stable updates of the weights. First, it bootstraps Q-values via temporal difference (TD) correction to train the current cirtic network. Due to off-policy nature, the stacked data in the replay buffer would have high variance, it leads to unstable learning. The target networks take account of small portion of the current networks as updates so that it achieves balanced improvements. In other words, the target networks correct TD-corrected values of different policies: it can reduce the intrinsic variance! 

<img src="images/ddpg/InvertedPendulum-v5_gam_099.jpg" alt="Alt Text" width="500"/>
