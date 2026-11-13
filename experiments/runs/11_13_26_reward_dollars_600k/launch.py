"""Dollar completions from the PBRS 120k checkpoint. No progress credit."""

import tools.training as T
from example.train import PPOTrainer, main

CKPT = "/root/Orbi.tar/experiments/runs/11_12_26_reward_pbrs_360k/model.zip"
RUN = "/root/Orbi.tar/experiments/runs/11_13_26_reward_dollars_240k"


class Run(T.Run):
    def __init__(self, **kwargs):
        super().__init__(
            load=CKPT,
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
    main()


if __name__ == "__main__":
    go()
