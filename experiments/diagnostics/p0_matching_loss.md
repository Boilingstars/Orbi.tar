# P0 matching-loss report

Source: `experiments/benchmarks/benchmark_20261117_155045/` on common event tapes, N=48, seeds 0–2 (preliminary). RL checkpoint is `ckpt_769024` (legend `pbrs_880k`). Missed USD is not treated as avoidable revenue: many due jobs cannot run together.

## Shift totals (mean of 3 seeds)

| Method | Revenue USD | Completed downlink | Completed relay | Downlink resets | DL cap util |
|---|---:|---:|---:|---:|---:|
| Hungarian + capacity repair | 7945 | 808 | 571 | 21 | 0.570 |
| RL ckpt_769024 | 7651 | 774 | 550 | 28 | 0.562 |
| Greedy | 7642 | 750 | 575 | 28 | 0.534 |

Paired Δ (RL 880k − Hungarian): **−$294 / shift (−3.7%)**. Per seed: −287, −298, −298.

## What the money gap is

1. **Fewer finished downlinks, not cheaper jobs.** Revenue per completed job is essentially the same (Hungarian 5.765, RL 5.779). q_selection is also the same (~1.03). The gap is volume: Hungarian finishes **+34 downlinks** and **+20 relays**. Downlink USD accounts for most of the $294.
2. **Downlink continuity.** RL resets ~8 more downlinks per shift than Hungarian (28 vs 21). Each reset restarts work. That is a confirmed operational loss, not a calibration penalty.
3. **Sequential decoder vs one-tick global V/T.** Hungarian maximises `V_j / T_ij` over the whole sat–slot table, then repairs the downlink cap. RL walks jobs in learned score order and assigns greedily. Hungarian still needs repair on ~34 ticks at N=48, so the cap is binding; RL already respects the cap while matching, which can leave a feasible high-ρ downlink unassigned after an earlier low-ρ take.
4. **Score-order PPO replay.** `_match()` uses `job_priority.argsort()`. After a weight update the same joint assignment is factored in a different job order, so `log π(a|s)` at replay is not the sampling factorisation. Packed `decoder_v2` removes that. This is a likelihood bug, not itself a dollar measurement.
5. **Observation saturation.** v1 `job_worth / 4` is clipped to 5. A p3 job at $18 and one at $40 look identical in that channel. Hungarian sees raw `value_usd / T_ij`.
6. **Physics still dominates leftover misses.** Even Hungarian misses ~1027 downlinks of ~1835 due. Extra RL misses are a thin slice of that infeasible remainder. Do not treat missed_value_usd as recoverable.

## What is not the main gap

- Greedy vs RL 880k is ~$9. The Hungarian gap is an order of magnitude larger.
- RL 500k (`ckpt_481280`) is worse on revenue (−$542 vs Hungarian) despite more completions than 880k: more jobs, cheaper mix. Completions without revenue are not an improvement.
- Calibration penalties were not used as an explanation and were not added.

## Per-tick diagnostics

`--diagnose` records ρ, quantiles, skip rate, unused capacity with a feasible leftover downlink, and Hungarian pair disagreement on the same state. It is off by default. A full N=48 diagnose was not run here (Hungarian is solved on every RL tick). Use it on a short smoke before claiming a causal matching defect.

Short counterfactual rollouts (5/10/20 ticks) were not added: copying the env, RNG, and event tape is not a reliable causal instrument yet. Descriptive metrics above are the P0 evidence.

## Training implication

The dollar gap is **downlink throughput and continuity**, with a **V/T-blind, order-dependent decoder** on top of a **clipped price channel**. Next spend should test one of those, not a reward rewrite and not imitation of Hungarian actions.
