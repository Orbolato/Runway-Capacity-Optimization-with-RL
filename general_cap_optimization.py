"""Train the Q(lambda) runway-capacity agent from the 2023 SITRAER paper.

The default parameters reproduce the setup reported in Section 4 of:
Carvalho, Murça, and Guterres (2023), "A Reinforcement Learning Approach for
Optimizing the Runway Capacity Utilization Under Uncertainty."
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import capoptimizer as cp

X_VMC = [0, 17, 24, 28, 28]
Y_VMC = [30, 30, 24, 15, 0]
X_IMC = [0, 12, 17, 20, 20]
Y_IMC = [22, 22, 17, 11, 0]

PAPER_EPISODES = 2_000_000
PAPER_STEP_SIZE = 0.5
PAPER_DISCOUNT_RATE = 0.9
PAPER_TRACE_DECAY = 0.5
PAPER_INITIAL_EPSILON = 1.0
PAPER_FINAL_EPSILON = 1e-6
PAPER_EPSILON_DECAY_EPISODES = 1_000_000
PAPER_ARRIVAL_PRIORITY = 2.0


def build_capacity_envelopes() -> Tuple[Dict[int, List[int]], Dict[int, List[float]]]:
    """Build the discrete VMC/IMC action envelopes used by the tabular agent."""
    u_vmc = list(range(max(X_VMC) + 1))
    u_imc = list(range(max(X_IMC) + 1))
    u = {0: u_vmc, 1: u_imc}
    v = {
        0: cp.f(X_VMC, Y_VMC, u_vmc),
        1: cp.f(X_IMC, Y_IMC, u_imc),
    }
    return u, v


def load_demand(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the demand/probability series from the paper dataset."""
    data = pd.read_excel(path, index_col=0)
    required = ["Arrival Demand", "Departure Demand", "IMC Probability"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    arrivals = data["Arrival Demand"].to_numpy(dtype=int)
    departures = data["Departure Demand"].to_numpy(dtype=int)
    imc_probability = data["IMC Probability"].to_numpy(dtype=float)

    if np.any(arrivals < 0) or np.any(departures < 0):
        raise ValueError("arrival and departure demands must be nonnegative")
    if np.any((imc_probability < 0) | (imc_probability > 1)):
        raise ValueError("IMC probabilities must be in [0, 1]")

    return arrivals, departures, imc_probability


def plot_capacity_envelopes() -> None:
    plt.figure()
    plt.plot(X_VMC, Y_VMC, label="VMC")
    plt.plot(X_IMC, Y_IMC, label="IMC")
    plt.xlabel("Arrivals")
    plt.ylabel("Departures")
    plt.legend()
    plt.tight_layout()


def train(
    arrival_demand: np.ndarray,
    departure_demand: np.ndarray,
    imc_probability: np.ndarray,
    *,
    arrival_priority: float = PAPER_ARRIVAL_PRIORITY,
    initial_arrival_queue: int = 0,
    initial_departure_queue: int = 0,
    episodes: int = PAPER_EPISODES,
    step_size: float = PAPER_STEP_SIZE,
    discount_rate: float = PAPER_DISCOUNT_RATE,
    trace_decay: float = PAPER_TRACE_DECAY,
    initial_epsilon: float = PAPER_INITIAL_EPSILON,
    final_epsilon: float = PAPER_FINAL_EPSILON,
    epsilon_decay_episodes: int = PAPER_EPSILON_DECAY_EPISODES,
    report_every: int = 10_000,
    history_every: int = 1_000,
    seed: Optional[int] = None,
) -> Tuple[cp.EligibilityTraceAgent, np.ndarray, np.ndarray, np.ndarray]:
    """Train the paper's dynamically stored tabular Q(lambda) agent."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if report_every <= 0 or history_every <= 0:
        raise ValueError("report_every and history_every must be positive")
    if not (
        len(arrival_demand) == len(departure_demand) == len(imc_probability)
    ):
        raise ValueError("demand and probability arrays must have equal length")
    if len(arrival_demand) == 0:
        raise ValueError("the demand horizon cannot be empty")

    if seed is not None:
        np.random.seed(seed)

    u, v = build_capacity_envelopes()
    env = cp.RunwayConfigEnv(
        arrival_priority=arrival_priority,
        u=u,
        v=v,
        reward_func="linear",
    )
    agent = cp.EligibilityTraceAgent(
        step_size=step_size,
        discount_rate=discount_rate,
        initial_epsilon=initial_epsilon,
        final_epsilon=final_epsilon,
        decaying_steps=epsilon_decay_episodes,
        et_decay=trace_decay,
        epsilon_decay="exponential",
        ucb=0.0,
    )

    history_episode: List[int] = []
    history_avg_cost: List[float] = []
    history_avg_return: List[float] = []
    horizon = len(arrival_demand)

    for episode in range(episodes):
        arrivals_queue = int(initial_arrival_queue + arrival_demand[0])
        departures_queue = int(initial_departure_queue + departure_demand[0])

        weather_condition = int(np.random.random() < imc_probability[0])
        env.set_state(weather_condition, arrivals_queue, departures_queue, 0)
        agent.check_state(env)
        action_index, action, reward = agent.take_epsilon_greedy_action(env, episode)

        for t in range(1, horizon):
            current_state = env.get_state()
            current_action_index = int(action_index)
            current_reward = float(reward)

            arrivals_queue += int(arrival_demand[t]) - int(action[0])
            departures_queue += int(departure_demand[t]) - int(action[1])

            weather_condition = int(np.random.random() < imc_probability[t])
            env.set_state(weather_condition, arrivals_queue, departures_queue, t)
            agent.check_state(env)
            action_index, action, reward = agent.take_epsilon_greedy_action(env, episode)

            agent.policy_eval(
                current_reward,
                current_state,
                current_action_index,
                env.get_state(),
                int(action_index),
            )

        agent.policy_eval(float(reward), env.get_state(), int(action_index), None, None)

        cost, _ = agent.get_return()
        avg_cost, avg_return, _, _ = agent.compute_avg_return(1.0, episode)

        if episode == 0 or (episode + 1) % history_every == 0 or episode + 1 == episodes:
            history_episode.append(episode + 1)
            history_avg_cost.append(avg_cost)
            history_avg_return.append(avg_return)

        if episode == 0 or (episode + 1) % report_every == 0 or episode + 1 == episodes:
            print(
                f"Episode {episode + 1:,}/{episodes:,}: "
                f"cost={cost:.2f}, average cost={avg_cost:.2f}, "
                f"average return={avg_return:.2f}, epsilon={agent.epsilon:.3e}"
            )

        agent.new_episode(episode)

    return (
        agent,
        np.asarray(history_episode),
        np.asarray(history_avg_cost),
        np.asarray(history_avg_return),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Q(lambda) runway-capacity model from the 2023 SITRAER paper."
    )
    parser.add_argument("--data", type=Path, default=Path("demand.xlsx"))
    parser.add_argument("--episodes", type=int, default=PAPER_EPISODES)
    parser.add_argument("--arrival-priority", type=float, default=PAPER_ARRIVAL_PRIORITY)
    parser.add_argument("--initial-arrival-queue", type=int, default=0)
    parser.add_argument("--initial-departure-queue", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=PAPER_STEP_SIZE)
    parser.add_argument("--discount-rate", type=float, default=PAPER_DISCOUNT_RATE)
    parser.add_argument("--trace-decay", type=float, default=PAPER_TRACE_DECAY)
    parser.add_argument("--initial-epsilon", type=float, default=PAPER_INITIAL_EPSILON)
    parser.add_argument("--final-epsilon", type=float, default=PAPER_FINAL_EPSILON)
    parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=PAPER_EPSILON_DECAY_EPISODES,
    )
    parser.add_argument("--report-every", type=int, default=10_000)
    parser.add_argument("--history-every", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="display the capacity envelopes and sampled learning curves",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrival_demand, departure_demand, imc_probability = load_demand(args.data)

    if args.show_plots:
        plot_capacity_envelopes()

    agent, x_history, avg_cost, avg_return = train(
        arrival_demand,
        departure_demand,
        imc_probability,
        arrival_priority=args.arrival_priority,
        initial_arrival_queue=args.initial_arrival_queue,
        initial_departure_queue=args.initial_departure_queue,
        episodes=args.episodes,
        step_size=args.step_size,
        discount_rate=args.discount_rate,
        trace_decay=args.trace_decay,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        report_every=args.report_every,
        history_every=args.history_every,
        seed=args.seed,
    )

    print(f"Visited states: {len(agent.states):,}")
    print(f"Final running-average cost: {agent.avg_cost:.2f}")

    if args.show_plots:
        plt.figure()
        plt.plot(x_history, avg_cost)
        plt.xlabel("Episodes")
        plt.ylabel("Average delay/cost")
        plt.tight_layout()

        plt.figure()
        plt.plot(x_history, avg_return)
        plt.xlabel("Episodes")
        plt.ylabel("Average return")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
