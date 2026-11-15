"""PBRS from scratch on P03, 240k steps, no warm-start checkpoint."""

import tools.training as T
from example.train import PPOTrainer, main

RUN = "/root/Orbi.tar/experiments/runs/11_14_26_reward_pbrs_240k"


class Run(T.Run):
    def __init__(self, **kwargs):
        super().__init__(
            timesteps=240_000,
            run_dir=RUN,
            n_envs=8,
            subproc=True,
            events=True,
            scenario="p03",
            top_k=64,
            **kwargs,
        )


def go() -> None:
    T.Run = Run
    PPOTrainer.ent_coef = 0.002
    PPOTrainer.learning_rate = 3e-5
    PPOTrainer.checkpoint_max_metric = "rollout/ep_rew_mean"
    PPOTrainer.checkpoint_every_n_max = 1
    PPOTrainer.plot_on_log = True
    main()


if __name__ == "__main__":
    go()
