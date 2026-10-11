"""Nonlinear-programming benchmark used in the 2023 SITRAER paper.

The optimization model matches the paper's fixed-runway-configuration benchmark:
for each time slot and sampled weather condition, integer arrival/departure
service rates minimize the same quadratic delay/cost used by the RL agent,
subject to the corresponding runway-capacity envelope and queue bounds.

Compared with the original script, the AMPL model is built once and reused for
all time slots. This preserves the optimization problem while removing a large
amount of avoidable interpreter/model-construction overhead.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X_VMC = [0, 17, 24, 28, 28]
Y_VMC = [30, 30, 24, 15, 0]
X_IMC = [0, 12, 17, 20, 20]
Y_IMC = [22, 22, 17, 11, 0]

PAPER_EPISODES = 1_000
PAPER_ARRIVAL_PRIORITY = 2.0


def load_demand(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def capacity_constraints(
    x: Sequence[float], y: Sequence[float]
) -> Tuple[List[float], List[float], List[float]]:
    """Return p, q, r for envelope inequalities p*v + q*u <= r."""
    if len(x) != len(y) or len(x) < 4:
        raise ValueError("invalid capacity envelope")

    p = [1.0]
    q = [0.0]
    r = [float(max(y))]

    for i in range(1, len(x) - 2):
        dx = x[i + 1] - x[i]
        if dx == 0:
            raise ValueError("unexpected vertical segment inside the envelope")
        slope = (y[i + 1] - y[i]) / dx
        p.append(1.0)
        q.append(float(-slope))
        r.append(float(y[i] - slope * x[i]))

    p.append(0.0)
    q.append(1.0)
    r.append(float(max(x)))
    return p, q, r


def build_ampl(ampl_path: Optional[Path], solver: str, solver_options: str):
    """Create and initialize the reusable AMPL optimization model."""
    try:
        from amplpy import AMPL, Environment
    except ImportError as exc:
        raise RuntimeError(
            "amplpy is required for this benchmark. Install requirements-ampl.txt "
            "and ensure AMPL plus the selected solver are available."
        ) from exc

    ampl = AMPL(Environment(str(ampl_path))) if ampl_path else AMPL()
    ampl.eval(
        r"""
        param a_t >= 0;
        param d_t >= 0;
        param alpha > 0;
        param p{i in 1..4};
        param q{i in 1..4};
        param r{i in 1..4};

        var u integer >= 0;
        var v integer >= 0;

        minimize delay:
            alpha*(a_t - u)^2 + (d_t - v)^2;

        s.t. capacity{i in 1..4}:
            p[i]*v + q[i]*u <= r[i];
        s.t. arrivals_available:
            u <= a_t;
        s.t. departures_available:
            v <= d_t;
        """
    )
    ampl.option["solver"] = solver
    if solver_options:
        ampl.option[f"{solver}_options"] = solver_options
    return ampl


def solve_benchmark(
    arrival_demand: np.ndarray,
    departure_demand: np.ndarray,
    imc_probability: np.ndarray,
    *,
    episodes: int = PAPER_EPISODES,
    arrival_priority: float = PAPER_ARRIVAL_PRIORITY,
    initial_arrival_queue: int = 0,
    initial_departure_queue: int = 0,
    ampl_path: Optional[Path] = None,
    solver: str = "knitro",
    solver_options: str = "ms_enable=1 ms_num_to_save=5 ms_savetol=0.01 ms_maxsolves=10",
    report_every: int = 10,
    seed: Optional[int] = None,
) -> np.ndarray:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if report_every <= 0:
        raise ValueError("report_every must be positive")
    if not (
        len(arrival_demand) == len(departure_demand) == len(imc_probability)
    ):
        raise ValueError("demand and probability arrays must have equal length")

    if seed is not None:
        np.random.seed(seed)

    constraints: Dict[int, Tuple[List[float], List[float], List[float]]] = {
        0: capacity_constraints(X_VMC, Y_VMC),
        1: capacity_constraints(X_IMC, Y_IMC),
    }

    ampl = build_ampl(ampl_path, solver, solver_options)
    ampl.param["alpha"] = float(arrival_priority)

    horizon = len(arrival_demand)
    episode_costs = np.empty(episodes, dtype=float)

    for episode in range(episodes):
        arrivals_queue = int(initial_arrival_queue)
        departures_queue = int(initial_departure_queue)
        total_cost = 0.0

        for t in range(horizon):
            arrivals_queue += int(arrival_demand[t])
            departures_queue += int(departure_demand[t])
            weather_condition = int(np.random.random() < imc_probability[t])
            p, q, r = constraints[weather_condition]

            ampl.param["a_t"] = arrivals_queue
            ampl.param["d_t"] = departures_queue
            ampl.param["p"] = p
            ampl.param["q"] = q
            ampl.param["r"] = r
            ampl.solve()

            solve_result = str(ampl.get_value("solve_result"))
            if solve_result != "solved":
                raise RuntimeError(
                    f"AMPL failed at episode {episode + 1}, slot {t + 1}: {solve_result}"
                )

            slot_cost = float(
                ampl.get_value("alpha*(a_t - u)^2 + (d_t - v)^2")
            )
            arrivals_served = int(round(ampl.get_value("u")))
            departures_served = int(round(ampl.get_value("v")))

            arrivals_queue -= arrivals_served
            departures_queue -= departures_served
            total_cost += slot_cost

        episode_costs[episode] = total_cost

        if episode == 0 or (episode + 1) % report_every == 0 or episode + 1 == episodes:
            print(
                f"Episode {episode + 1:,}/{episodes:,}: "
                f"cost={total_cost:.2f}, mean={episode_costs[: episode + 1].mean():.2f}"
            )

    return episode_costs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AMPL/Knitro benchmark from the 2023 SITRAER paper."
    )
    parser.add_argument("--data", type=Path, default=Path("demand.xlsx"))
    parser.add_argument("--episodes", type=int, default=PAPER_EPISODES)
    parser.add_argument("--arrival-priority", type=float, default=PAPER_ARRIVAL_PRIORITY)
    parser.add_argument("--initial-arrival-queue", type=int, default=0)
    parser.add_argument("--initial-departure-queue", type=int, default=0)
    parser.add_argument(
        "--ampl-path",
        type=Path,
        default=None,
        help="optional AMPL installation directory; omit if AMPL is on PATH",
    )
    parser.add_argument("--solver", default="knitro")
    parser.add_argument(
        "--solver-options",
        default="ms_enable=1 ms_num_to_save=5 ms_savetol=0.01 ms_maxsolves=10",
    )
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrival_demand, departure_demand, imc_probability = load_demand(args.data)
    costs = solve_benchmark(
        arrival_demand,
        departure_demand,
        imc_probability,
        episodes=args.episodes,
        arrival_priority=args.arrival_priority,
        initial_arrival_queue=args.initial_arrival_queue,
        initial_departure_queue=args.initial_departure_queue,
        ampl_path=args.ampl_path,
        solver=args.solver,
        solver_options=args.solver_options,
        report_every=args.report_every,
        seed=args.seed,
    )

    print(f"Average cost over {len(costs):,} episodes: {costs.mean():.2f}")

    if args.show_plot:
        x = np.arange(1, len(costs) + 1)
        plt.figure()
        plt.plot(x, costs, "o", markersize=3, label="Episode cost")
        plt.axhline(costs.mean(), linestyle="--", label="Mean cost")
        plt.xlabel("Episodes")
        plt.ylabel("Delay/Cost")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
