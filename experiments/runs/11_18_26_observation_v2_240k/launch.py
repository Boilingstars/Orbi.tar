"""Fine-tune the screened 880k dispatcher with observation_v2 only.

Weights from ckpt_769024, widened v1->v2 (new channels start at zero).
Decoder and reward stay v1 / current_pbrs. Adam is fresh. Same PPO
hyperparameters as the 880k PBRS run.
"""

import os

import tools.training as T
from example.train import PPOTrainer, main

CKPT = "/root/Orbi.tar/experiments/runs/11_15_26_reward_pbrs_880k/checkpoints/ckpt_769024.zip"
RUN = "/root/Orbi.tar/experiments/runs/11_18_26_observation_v2_240k"
SMOKE = os.environ.get("ORBI_SMOKE") == "1"


class Run(T.Run):
    def __init__(self, **kwargs):
        super().__init__(
            load=CKPT,
            resume=False,
            optimizer_resume_mode="weights_only",
            migrate_observation=True,
            timesteps=1_024 if SMOKE else 240_000,
            run_dir=RUN + "_smoke" if SMOKE else RUN,
            n_envs=8,
            subproc=True,
            events=True,
            scenario="p03",
            top_k=64,
            seed=0,
            progress_bar=False,
            observation_version="v2",
            decoder_version="v1",
            reward_mode="current_pbrs",
            experiment_id="observation_v2_240k_smoke" if SMOKE else "observation_v2_240k",
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
    PPOTrainer.reward_mode = "current_pbrs"
    PPOTrainer.observation_version = "v2"
    PPOTrainer.decoder_version = "v1"
    PPOTrainer.optimizer_resume_mode = "weights_only"
    PPOTrainer.migrate_observation = True
    PPOTrainer.experiment_id = "observation_v2_240k_smoke" if SMOKE else "observation_v2_240k"
    PPOTrainer.checkpoint_max_metric = "rollout/ep_rew_mean"
    PPOTrainer.checkpoint_every_n_max = 1
    PPOTrainer.plot_on_log = True
    main()


if __name__ == "__main__":
    go()
