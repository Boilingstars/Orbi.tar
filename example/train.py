"""Train this package's agents. Launch settings are `Run` in `tools.training`.

`TRAINING` pairs an agent from `example.agent` with a function in this file.
"""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import example.jobs
from example.agent import Ctde, Dispatcher, GreedyAgent
from example.models import OpsAttentionPolicy, count_parameters
from example.plots import plot_run
from orbi import load, resolve_scenario
from orbi.env import N_MAX, EventConfig, OpsEnv
from tools.evaluating import load_ppo
from tools.training import Session, Trainer, launch


# Width of the network. `light` is the same layout with fewer weights.
PROFILES = {
    "full": dict(d=128, heads=4, blocks=2, ff=512, d_pointer=64),
    "light": dict(d=64, heads=4, blocks=1, ff=128, d_pointer=32),
}


TRAINING = (
    (Dispatcher, "maskable_ppo"),
    # (Dispatcher, "maskable_ppo_greedy"),
    (Ctde, "mappo"),
)

# Greedy imitation length, then the entropy bonus slides from START to END.
GREEDY_EPISODES = 4
ENTROPY_START = 0.03
ENTROPY_END = 0.001


# Fields copied from PPOTrainer into MaskablePPO. `n_steps` is the episode length.
_ALGO = (
    "learning_rate",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
)


class PPOTrainer(Trainer):
    """Step size, clip, and epoch count shared by every function below."""

    suite = "example"
    policy_name = "dispatcher"

    learning_rate = 1e-4
    value_learning_rate = 1e-4
    ent_coef = 0.03
    target_kl = 1.5
    batch_size = 128
    n_epochs = 4
    gamma = 0.995
    gae_lambda = 0.95
    clip_range = 0.2
    vf_coef = 0.5
    max_grad_norm = 0.5


class _JointQ(nn.Module):
    """Team return of a joint action. Swapping one satellite does not rerun the encoder."""

    def __init__(self, d: int, n_actions: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_actions, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def replacements(self, h_s: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.long()
        _, n, d = h_s.shape
        categories = self.embed.num_embeddings

        # Sum of (satellite state + the action it took).
        emb = self.embed(actions)
        base = (h_s + emb).sum(1)

        # For satellite i and category c, swap only that action and score the fleet.
        delta = self.embed.weight.view(1, 1, categories, d) - emb.unsqueeze(2)
        mixed = (base[:, None, None, :] + delta) / n
        return self.head(mixed).squeeze(-1)


class Mappo(MaskablePPO):
    """On-board sequential actor, fleet critic, a separate advantage per satellite."""

    def __init__(self, *args, **kwargs):
        kwargs["rollout_buffer_class"] = _MappoBuffer
        super().__init__(*args, **kwargs)

        # Q reads the actor width and the number of categories (slot, idle, calibrate).
        width = self.policy.net.actor.encoder.emb_sat[0].out_features
        self.q_head = _JointQ(width, int(self.action_space.nvec[0])).to(self.device)
        self._remember_agent_log_prob()

    def _excluded_save_params(self) -> list[str]:
        # Q is only used while training. The saved actor and critic still load for play.
        return super()._excluded_save_params() + ["q_head"]

    def _remember_agent_log_prob(self) -> None:
        policy = self.policy
        raw_forward = policy.forward
        raw_add = self.rollout_buffer.add

        def forward(obs, deterministic=False, action_masks=None):
            actions, values, joint = raw_forward(obs, deterministic=deterministic, action_masks=action_masks)

            # Per-satellite log-probability of the action just sampled.
            with torch.no_grad():
                per, _, _ = _actor_terms(policy.net, obs, actions, action_masks)
            policy._last_agent_logp = per.detach().cpu().numpy()
            return actions, values, joint

        def add(*args, **kwargs):
            # The buffer is about to write this step at `pos`.
            per = getattr(policy, "_last_agent_logp", None)
            if per is not None:
                self.rollout_buffer.agent_log_probs[self.rollout_buffer.pos] = per
            return raw_add(*args, **kwargs)

        policy.forward = forward
        self.rollout_buffer.add = add

    def _loss(self, batch, clip_range):
        actions = batch.actions.long()
        obs, masks = batch.observations, batch.action_masks
        net = self.policy.net

        # Fleet state, and the decoder log-probability of the stored actions.
        h_s = net.critic(obs)[0]
        new_logp, rows, entropy = _actor_terms(net, obs, actions, masks)

        # Fit Q to the team return. The advantage uses Q but does not train it.
        q_all = self.q_head.replacements(h_s, actions)
        q_taken = q_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        returns = batch.returns.flatten()
        q_loss = F.mse_loss(q_taken.mean(-1), returns)
        adv = _counterfactual(q_all, q_taken, rows, self.normalize_advantage)

        # Clipped PPO, one ratio per satellite, plus the centralized value fit.
        value_loss = F.mse_loss(returns, self.policy.predict_values(obs).flatten())
        ratio = torch.exp(new_logp - batch.agent_log_prob)
        unclipped = adv * ratio
        clipped = adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(unclipped, clipped).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * (value_loss + q_loss)

        # Sum over satellites, so the KL cap stays the joint-policy cap.
        with torch.no_grad():
            log_ratio = new_logp - batch.agent_log_prob
            approx_kl = float(((torch.exp(log_ratio) - 1) - log_ratio).sum(-1).mean())
            clip_fraction = float((torch.abs(ratio - 1) > clip_range).float().mean())
        stats = {
            "pg": float(policy_loss.detach()),
            "value": float(value_loss.detach()),
            "q": float(q_loss.detach()),
            "entropy": float(entropy_loss.detach()),
            "clip": clip_fraction,
            "kl": approx_kl,
        }
        return loss, stats

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)

        # KL is reset each epoch. The other lists cover the whole update.
        buckets = {key: [] for key in ("pg", "value", "q", "entropy", "clip", "kl")}
        continue_training = True

        for _epoch in range(self.n_epochs):
            buckets["kl"] = []

            for batch in self.rollout_buffer.get(self.batch_size):
                loss, stats = self._loss(batch, clip_range)
                
                for key, value in stats.items():
                    buckets[key].append(value)

                # Stop when the summed per-satellite KL passes the PPO cap.
                if self.target_kl is not None and stats["kl"] > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                grads = list(self.policy.parameters()) + list(self.q_head.parameters())
                torch.nn.utils.clip_grad_norm_(grads, self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        # Same names MaskablePPO logs, plus the fit of the counterfactual Q.
        explained = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        logged = {
            "train/entropy_loss": np.mean(buckets["entropy"]),
            "train/policy_gradient_loss": np.mean(buckets["pg"]),
            "train/value_loss": np.mean(buckets["value"]),
            "train/q_loss": np.mean(buckets["q"]),
            "train/approx_kl": np.mean(buckets["kl"]),
            "train/clip_fraction": np.mean(buckets["clip"]),
            "train/explained_variance": explained,
            "train/clip_range": clip_range,
        }
        for key, value in logged.items():
            self.logger.record(key, value)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")


class _AnnealedPPO(MaskablePPO):
    """Same update as MaskablePPO. The entropy bonus falls across the run."""
    def train(self) -> None:
        # 0 at the first rollout, 1 when the step budget is spent.
        done = 1.0 - float(self._current_progress_remaining)
        self.ent_coef = ENTROPY_START + (ENTROPY_END - ENTROPY_START) * done

        super().train()
        self.logger.record("train/ent_coef", self.ent_coef)


def maskable_ppo(session: Session, agent) -> MaskablePPO:
    """RL from the first step. One team advantage for the joint action."""
    _only(agent, "dispatcher", "maskable_ppo")
    return _build(session, agent, MaskablePPO)


def maskable_ppo_greedy(session: Session, agent) -> MaskablePPO:
    """Copy greedy with cross-entropy, then RL with a fading entropy bonus."""
    _only(agent, "dispatcher", "maskable_ppo_greedy")
    model = _build(session, agent, _AnnealedPPO)

    # A loaded checkpoint has already left the teacher. Don't imitate again.
    episodes = 0 if session.run.load else GREEDY_EPISODES
    if episodes:
        _imitate_greedy(model, episodes)

    session.notes.update(
        method="maskable_ppo_greedy",
        greedy_episodes=episodes,
        entropy_start=ENTROPY_START,
        entropy_end=ENTROPY_END,
    )
    return model


def mappo(session: Session, agent) -> Mappo:
    """Central critic, local sequential actor, counterfactual advantage per satellite."""
    _only(agent, "ctde", "mappo")
    model = _build(session, agent, Mappo)
    session.notes["method"] = "mappo"
    return model


class _Launch(PPOTrainer):
    """Keeps the agent name. `launch` would otherwise label every run with `Run.controller`."""

    def __init__(self, agent, method_name: str) -> None:
        name = agent.controller
        if method_name == "maskable_ppo_greedy":
            name = f"{name}_greedy"
        self._policy_name = name

    @property
    def policy_name(self) -> str:
        return self._policy_name

    @policy_name.setter
    def policy_name(self, value: str) -> None:
        return


def main() -> None:
    for agent, method_name in TRAINING:
        method = globals()[method_name]
        launch(
            lambda session, agent=agent, method=method: method(session, agent),
            _Launch(agent, method_name),
            plot_run,
        )


def mask_fn(env) -> np.ndarray:
    return env.unwrapped.action_masks()


def _make_env(run, spec: dict, events: EventConfig, seed: int):
    def _thunk():
        env = OpsEnv(
            scenario=spec,
            events=events,
            top_k=run.top_k,
            max_steps=run.max_steps,
            seed=seed,
            baseline_repair=run.baseline_repair,
            neighbor_method=run.neighbor_method,
            neighbor_memory=run.neighbor_memory,
            ground_points=run.ground_points,
            far_job_p=run.far_job_p,
        )
        return ActionMasker(Monitor(env), mask_fn)

    return _thunk


def _only(agent, controller: str, method: str) -> None:
    if agent.controller != controller:
        raise ValueError(f"{method} trains the {controller}")


def _split_value_learning_rate(model: MaskablePPO, policy_lr: float, value_lr: float, extra_value=()) -> None:
    """Two Adam groups. SB3 resets every group to the policy step each rollout."""
    net = model.policy.net

    # Critic weights. MAPPO's joint-Q head trains with them.
    if hasattr(net, "critic"):
        value_params = list(net.critic.parameters())
    else:
        value_params = [
            p
            for module in (net.value_head, net.pool_sat, net.pool_job, net.pool_plan)
            for p in module.parameters()
        ]
    for module in extra_value:
        value_params.extend(module.parameters())

    # The actor is everything else.
    value_ids = {id(p) for p in value_params}
    policy_params = [p for p in net.parameters() if id(p) not in value_ids]
    model.policy.optimizer = model.policy.optimizer_class(
        [
            {"params": policy_params, "lr": policy_lr},
            {"params": value_params, "lr": value_lr},
        ],
        **model.policy.optimizer_kwargs,
    )

    # After each rollout SB3 writes the policy rate into both groups. Put the critic rate back.
    scale = value_lr / policy_lr
    original = model._update_learning_rate

    def _update(optimizers) -> None:
        original(optimizers)
        optimizer = model.policy.optimizer
        base = optimizer.param_groups[0]["lr"]
        optimizer.param_groups[1]["lr"] = base * scale

    def _excluded_save_params() -> list[str]:
        names = type(model)._excluded_save_params(model)
        return list(names) + ["_update_learning_rate", "_excluded_save_params"]

    model._update_learning_rate = _update
    model._excluded_save_params = _excluded_save_params


def _build(session: Session, agent, algo) -> MaskablePPO:
    """One vec-env and one policy. The reward is `MyReward`."""
    run = session.run
    trainer = session.trainer
    path = resolve_scenario(run.scenario)
    spec = load(path)

    # One wrapped environment per worker. Events follow `Run`.
    events = EventConfig() if run.events else EventConfig.off()
    thunks = [_make_env(run, spec, events, run.seed + i) for i in range(run.n_envs)]
    vec = SubprocVecEnv if run.subproc and run.n_envs > 1 else DummyVecEnv
    env = vec(thunks)

    # Same hyperparameters as PPOTrainer. A rollout is one episode.
    n_steps = run.max_steps or spec["time"]["steps"]
    policy_kwargs = dict(PROFILES[run.profile], controller=agent.controller)
    model = algo(
        OpsAttentionPolicy,
        env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        verbose=1,
        seed=run.seed,
        **{name: getattr(trainer, name) for name in _ALGO},
    )

    # Critic learning rate. MAPPO also puts the Q head in that group.
    extra = (model.q_head,) if isinstance(model, Mappo) else ()
    _split_value_learning_rate(model, trainer.learning_rate, trainer.value_learning_rate, extra)

    # Optional warm start from a saved policy.
    if run.load:
        loaded = load_ppo(run.load)
        model.policy.load_state_dict(loaded.policy.state_dict())
        print(f"Starting weights: {run.load}", flush=True)

    session.notes.update(
        scenario=str(path),
        scenario_id=spec["meta"]["id"],
        n_satellites=len(spec["satellites"]),
        steps=spec["time"]["steps"],
        n_max=N_MAX,
        n_steps=n_steps,
        policy=policy_kwargs,
        policy_parameters=count_parameters(model.policy.net),
        method=algo.__name__,
        agent=agent.__name__,
    )
    model._horizon = n_steps
    return model


def _stack_obs(frames: list[dict]) -> dict:
    return {key: np.concatenate([frame[key] for frame in frames], axis=0) for key in frames[0]}


def _imitate_greedy(model: MaskablePPO, episodes: int) -> None:
    """Cross-entropy on greedy's own trajectory.

    The decoder is sequential, so the teacher's action already keeps a slot
    from being given to the next satellite.
    """
    env = model.env
    teachers = [GreedyAgent(limit) for limit in env.get_attr("downlink_limit")]
    parameters = [p for p in model.policy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(parameters, lr=model.learning_rate)

    # One episode per env, repeated `episodes` times. Update every PPO batch.
    horizon = int(getattr(model, "_horizon", 1))
    steps = max(episodes, 1) * horizon
    hold = max(model.batch_size // max(env.num_envs, 1), 1)
    obs = env.reset()
    batch_obs, batch_act = [], []
    model.policy.set_training_mode(True)

    for seen in range(1, steps + 1):
        # What greedy would do in each environment on this observation.
        actions = [
            teacher.act({key: np.asarray(value[i]) for key, value in obs.items()})
            for i, teacher in enumerate(teachers)
        ]
        batch_obs.append(obs)
        batch_act.append(np.stack(actions))

        # Negative log-probability of that action. Lower means a closer copy.
        if len(batch_obs) == hold or seen == steps:
            stacked = _stack_obs(batch_obs)
            act = np.concatenate(batch_act, axis=0)
            obs_t = obs_as_tensor(stacked, model.device)
            act_t = torch.as_tensor(act, device=model.device)
            _, log_prob, _ = model.policy.evaluate_actions(obs_t, act_t, stacked["mask"].reshape(len(act), -1))
            loss = -log_prob.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, model.max_grad_norm)
            opt.step()
            if seen == steps or seen % max(horizon, 1) == 0:
                print(f"imitation {seen}/{steps} cross-entropy {float(loss.detach()):.3f}", flush=True)
            batch_obs, batch_act = [], []

        # The next state is the one greedy actually enters.
        obs, _, _, _ = env.step(np.stack(actions))
    print("Greedy imitation done", flush=True)


def _actor_terms(net, obs, actions, masks):
    """Log-probability of each satellite under the sequential decoder, and its masked logits."""
    actions = actions.long()
    _, _, entropy, rows = net.actor._run(obs, masks, actions=actions)

    # One log-probability per satellite, conditioned on the earlier choices.
    chosen = torch.log_softmax(rows, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return chosen, rows, entropy


class _WithLogProb:
    """A maskable minibatch plus each satellite's log-probability from the rollout."""

    def __init__(self, batch, agent_log_prob: torch.Tensor) -> None:
        self._batch = batch
        self.agent_log_prob = agent_log_prob

    def __getattr__(self, name: str):
        return getattr(self._batch, name)


class _MappoBuffer(MaskableDictRolloutBuffer):
    """Same minibatches as the maskable buffer, with per-satellite log-probabilities added."""

    def reset(self) -> None:
        super().reset()
        width = len(self.action_space.nvec)
        self.agent_log_probs = np.zeros((self.buffer_size, self.n_envs, width), dtype=np.float32)

    def get(self, batch_size: int | None = None) -> Generator[_WithLogProb, None, None]:
        # Flatten once, in the same moment the parent flattens its own arrays.
        if not self.generator_ready:
            self.agent_log_probs = self.swap_and_flatten(self.agent_log_probs)

        yield from super().get(batch_size)

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> _WithLogProb:
        base = super()._get_samples(batch_inds, env)
        return _WithLogProb(base, self.to_torch(self.agent_log_probs[batch_inds]))


def _counterfactual(q_all, q_taken, rows, normalize: bool) -> torch.Tensor:
    """Q(taken) minus the decoder's average Q. Every other satellite stays put."""
    with torch.no_grad():
        pi = torch.softmax(rows, dim=-1)
        adv = q_taken - (pi * q_all).sum(-1)

        if normalize:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv


if __name__ == "__main__":
    main()
