"""Fine-tune the screened v1 dispatcher on dollar completions only.

Weights from the screening winner (Adam is fresh). Same PPO hyperparameters
as the 880k PBRS run. `gamma=0.998` is PPO's discount, not a PBRS term.
"""

import os

import tools.training as T
from example.train import PPOTrainer, main

# Screening (seeds 10–14, N=48 common): ckpt_769024 beat ckpt_1159544 by ~$22/shift.
CKPT = "/root/Orbi.tar/experiments/runs/11_15_26_reward_pbrs_880k/checkpoints/ckpt_769024.zip"
RUN = "/root/Orbi.tar/experiments/runs/11_18_26_reward_dollars_240k"
SMOKE = os.environ.get("ORBI_SMOKE") == "1"


class Run(T.Run):
    def __init__(self, **kwargs):
        super().__init__(
            load=CKPT,
            resume=False,
            optimizer_resume_mode="weights_only",
            timesteps=1_024 if SMOKE else 240_000,
            run_dir=RUN + "_smoke" if SMOKE else RUN,
            n_envs=8,
            subproc=True,
            events=True,
            scenario="p03",
            top_k=64,
            seed=0,
            progress_bar=False,
            observation_version="v1",
            decoder_version="v1",
            reward_mode="dollars",
            experiment_id="dollars_240k_smoke" if SMOKE else "dollars_240k",
            **kwargs,
        )


def go() -> None:
    T.Run = Run
    PPOTrainer.ent_coef = 0.002
    PPOTrainer.learning_rate = 3e-5
    PPOTrainer.value_learning_rate = 3e-4
    PPOTrainer.gamma = 0.998
    PPOTrainer.gae_lambda = 0.95
    PPOTrainer.target_kl = 0.1
    PPOTrainer.batch_size = 128
    PPOTrainer.n_epochs = 4
    PPOTrainer.clip_range = 0.2
    PPOTrainer.reward_mode = "dollars"
    PPOTrainer.observation_version = "v1"
    PPOTrainer.decoder_version = "v1"
    PPOTrainer.optimizer_resume_mode = "weights_only"
    PPOTrainer.experiment_id = "dollars_240k_smoke" if SMOKE else "dollars_240k"
    PPOTrainer.checkpoint_max_metric = "rollout/ep_rew_mean"
    PPOTrainer.checkpoint_every_n_max = 1
    PPOTrainer.plot_on_log = True
    main()


if __name__ == "__main__":
    go()
