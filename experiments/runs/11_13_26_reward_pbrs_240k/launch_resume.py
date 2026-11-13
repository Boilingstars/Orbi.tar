"""Resume the PBRS 240k run from the latest zip (weights + Adam). Stop the live trainer first."""

import tools.training as T
from example.train import PPOTrainer, main

CKPT = "/root/Orbi.tar/experiments/runs/11_13_26_reward_pbrs_240k/resume_latest.zip"
RUN = "/root/Orbi.tar/experiments/runs/11_13_26_reward_pbrs_240k"


class Run(T.Run):
    def __init__(self, **kwargs):
        super().__init__(
            load=CKPT,
            timesteps=240_000,
            run_dir=RUN,
            resume=True,
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
    PPOTrainer.plot_on_log = True
    PPOTrainer.plot_every_s = 20 * 60
    main()


if __name__ == "__main__":
    go()
