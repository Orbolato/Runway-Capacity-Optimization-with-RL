# Runway Capacity Optimization with Reinforcement Learning

Public implementation associated with the conference paper:

**Lucas Orbolato Carvalho, Mayara Condé Rocha Murça, and Marcelo Xavier Guterres, “A Reinforcement Learning Approach for Optimizing the Runway Capacity Utilization Under Uncertainty,” XX SITRAER – Air Transportation Symposium, Joinville, Brazil, 2023.**

- Conference record: https://conferencias.ufsc.br/index.php/sitraer/xx/paper/view/2181
- Full-text record: https://www.researchgate.net/publication/376480738_A_REINFORCEMENT_LEARNING_APPROACH_FOR_OPTIMIZING_THE_RUNWAY_CAPACITY_UTILIZATION_UNDER_UNCERTAINTY

> **Scope:** this repository corresponds to the **2023 SITRAER study**. It is not the implementation used in the later 2025 *Transportation Research Part A: Policy and Practice* paper on stochastic model-free runway-capacity management.

## What the code implements

The paper studies runway-capacity utilization for a **fixed runway configuration** under uncertain weather. The environment alternates stochastically between visual meteorological conditions (VMC) and instrument meteorological conditions (IMC), each with its own runway-configuration capacity envelope.

The published reinforcement-learning formulation is preserved in the code:

- state: `S_t = (weather condition, arrival queue, departure queue, time period)`;
- action: integer arrival/departure service rates `(u_t, v_t)` on or immediately below the active capacity envelope and bounded by the queues;
- reward: the negative quadratic delay/cost function used in the paper;
- method: tabular **Q(lambda)** with eligibility traces;
- exploration: exponentially decaying epsilon-greedy policy;
- storage: states and feasible actions are initialized dynamically when first visited.

`capoptimizer.py` also retains one-step SARSA and Q-learning agents for comparison, while `EligibilityTraceAgent` is the implementation corresponding to the paper's proposed method.

## Paper-faithful default parameters

The defaults in `general_cap_optimization.py` match the values reported in Section 4 of the paper:

| Parameter | Symbol | Value |
|---|---:|---:|
| Episodes | – | 2,000,000 |
| Step size | kappa | 0.5 |
| Discount rate | gamma | 0.9 |
| Eligibility-trace decay | lambda | 0.5 |
| Initial epsilon | epsilon_i | 1.0 |
| Final epsilon | epsilon_f | 1e-6 |
| Epsilon-decay horizon | – | 1,000,000 episodes |
| Arrival priority | alpha | 2.0 |

The previous public driver script used `step_size=0.9` and `final_epsilon=1e-16`, which did **not** match the parameter values stated in the published paper. Those defaults were corrected during the 2026 code cleanup; the underlying algorithm was not changed.

## Repository structure

```text
.
├── capoptimizer.py                    # Capacity envelope, environment, and tabular RL agents
├── general_cap_optimization.py        # Q(lambda) experiment from the paper
├── capacity_optimization_NLP_AMPL.py  # AMPL/Knitro nonlinear-programming benchmark
├── demand.xlsx                        # Demand and IMC-probability inputs used by the scripts
├── requirements.txt                   # Python dependencies for the RL experiment/tests
├── requirements-ampl.txt              # Optional AMPL Python dependency
└── tests/                              # Unit tests for the paper-specific implementation
```

## Main improvements in the cleaned implementation

The cleanup keeps the scientific formulation tied to the paper while improving the software implementation:

- removes mutable default arguments that could make separate agent instances share state accidentally;
- consolidates duplicated SARSA/Q-learning/Q(lambda) code into a common tabular-agent implementation;
- preserves the original public class names through backward-compatible aliases;
- validates states, parameters, demand columns, and probability ranges;
- uses explicit positional arrays instead of deprecated/ambiguous pandas indexing inside training loops;
- avoids printing and storing all 2,000,000 learning episodes by default while keeping the running average mathematically identical;
- adds command-line arguments so paper defaults can be reproduced without editing source code;
- adds an optional random seed for repeatable new runs (the paper itself did not report a fixed seed);
- builds the AMPL model once and reuses it instead of recreating AMPL at every time step;
- removes unnecessary per-episode plotting from the AMPL benchmark;
- adds automated tests and continuous integration.

The AMPL cleanup substantially reduces implementation overhead. Therefore, **new wall-clock timings from the cleaned AMPL script should not be interpreted as a reproduction of the paper's historical computational-time comparison**. The optimization model and objective/constraints remain the same.

## Installation

For the reinforcement-learning experiment and tests:

```bash
python -m pip install -r requirements.txt
```

For the optional AMPL benchmark:

```bash
python -m pip install -r requirements-ampl.txt
```

AMPL and the selected solver must also be installed/licensed separately.

## Run the reinforcement-learning experiment

The paper configuration is the default:

```bash
python general_cap_optimization.py
```

For a quick smoke test:

```bash
python general_cap_optimization.py --episodes 1000 --epsilon-decay-episodes 500 --seed 42
```

Show the capacity envelopes and sampled learning curves with:

```bash
python general_cap_optimization.py --show-plots
```

Use `python general_cap_optimization.py --help` to see all options.

### Output sampling

The running average cost and return are still updated **every episode**, exactly as required by the paper formulation. For efficiency, only sampled values are stored for plotting (`--history-every`, default 1000), and progress is printed periodically (`--report-every`, default 10000). These settings do not affect learning.

## Run the AMPL/Knitro benchmark

If AMPL is available on the system path:

```bash
python capacity_optimization_NLP_AMPL.py
```

If an explicit AMPL installation directory is required:

```bash
python capacity_optimization_NLP_AMPL.py --ampl-path /path/to/ampl
```

The paper uses 1000 stochastic benchmark episodes by default. Use `--show-plot` to display the episode-cost scatter and mean.

## Tests

```bash
python -m pytest -q
```

The tests cover the published capacity envelopes, dynamic feasible-action construction, reward/cost definition, Q(lambda) terminal update, epsilon decay, paper parameter defaults, and backward-compatible API aliases.

## Notes on interpretation

This repository intentionally keeps the 2023 study's scope:

- one fixed runway configuration;
- two weather-dependent capacity envelopes (VMC and IMC);
- integer arrival/departure service rates;
- weather uncertainty represented through time-dependent IMC probabilities;
- tabular state-action values and dynamic storage of visited states/actions.

Later research by the authors expands the runway-capacity-management problem substantially. Those later formulations should not be inferred from this repository.
