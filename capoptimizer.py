"""Core tabular reinforcement-learning components for runway-capacity allocation.

The implementation follows the 2023 SITRAER paper
"A Reinforcement Learning Approach for Optimizing the Runway Capacity
Utilization Under Uncertainty". The state, action, reward, dynamic state/action
storage, Q-learning, and Q(lambda) update rules are intentionally kept aligned
with Algorithm 1 in that paper.
"""

from __future__ import annotations

from typing import Iterable, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

State = Tuple[int, int, int, int]
Action = List[int]
StateAction = Tuple[int, int, int, int, int]


def f(x: Sequence[float], y: Sequence[float], u: Iterable[float]) -> List[float]:
    """Return the departure-capacity upper bound for each arrival rate.

    This is the runway-configuration capacity-envelope function used in the
    paper. ``x`` contains arrival coordinates of the envelope breakpoints,
    ``y`` contains the corresponding departure coordinates, and ``u`` contains
    arrival service rates at which the envelope is evaluated.

    The final vertical segment used in the paper (e.g. x=[..., 28, 28]) is
    handled exactly as in the original implementation: at the maximum arrival
    rate the upper endpoint of that vertical segment is returned.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of breakpoints")
    if len(x) < 2:
        raise ValueError("at least two capacity-envelope breakpoints are required")
    if any(x[i] > x[i + 1] for i in range(len(x) - 1)):
        raise ValueError("x breakpoints must be nondecreasing")

    upper_bounds: List[float] = []
    x_last = x[-1]

    for arrival_rate in u:
        if arrival_rate < 0 or arrival_rate > x_last:
            upper_bounds.append(0.0)
            continue

        if arrival_rate == x_last:
            dx = x[-1] - x[-2]
            upper_bounds.append(float(y[-2]) if dx == 0 else 0.0)
            continue

        value: Optional[float] = None
        for j in range(len(x) - 1):
            if x[j] <= arrival_rate < x[j + 1]:
                dx = x[j + 1] - x[j]
                if dx == 0:
                    value = float(y[j])
                else:
                    slope = (y[j + 1] - y[j]) / dx
                    value = float(slope * (arrival_rate - x[j]) + y[j])
                break

        upper_bounds.append(0.0 if value is None else value)

    return upper_bounds


capacity_envelope_departures = f


class _BaseAgent:
    """Shared implementation for the tabular agents used in the repository."""

    def __init__(
        self,
        states: Optional[MutableMapping[State, bool]] = None,
        actions: Optional[MutableMapping[State, List[Action]]] = None,
        q_values: Optional[MutableMapping[State, List[float]]] = None,
        N_a: Optional[MutableMapping[State, List[int]]] = None,
        episode: int = 0,
        step_size: float = 0.5,
        discount_rate: float = 0.9,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 1e-6,
        decaying_steps: int = 100000,
        epsilon_decay: str = "exponential",
        ucb: float = 0.1,
    ) -> None:
        del episode

        self.states = {} if states is None else states
        self.actions = {} if actions is None else actions
        self.q_values = {} if q_values is None else q_values
        self.N_a = {} if N_a is None else N_a

        if not 0 < step_size <= 1:
            raise ValueError("step_size must be in (0, 1]")
        if not 0 <= discount_rate <= 1:
            raise ValueError("discount_rate must be in [0, 1]")
        if not 0 <= final_epsilon <= initial_epsilon <= 1:
            raise ValueError("epsilon values must satisfy 0 <= final <= initial <= 1")
        if decaying_steps <= 0:
            raise ValueError("decaying_steps must be positive")
        if epsilon_decay not in {"exponential", "linear"}:
            raise ValueError("epsilon_decay must be 'exponential' or 'linear'")
        if epsilon_decay == "exponential" and final_epsilon <= 0:
            raise ValueError("final_epsilon must be positive for exponential decay")
        if ucb < 0:
            raise ValueError("ucb must be nonnegative")

        self.step_size = float(step_size)
        self.discount_rate = float(discount_rate)
        self.initial_epsilon = float(initial_epsilon)
        self.epsilon = float(initial_epsilon)
        self.final_epsilon = float(final_epsilon)
        self.decaying_steps = int(decaying_steps)
        self.epsilon_decay = epsilon_decay
        self.ucb = float(ucb)

        self.cost = 0.0
        self.G = 0.0
        self.avg_cost = 0.0
        self.avg_G = 0.0

    @staticmethod
    def _initial_q_value(env: "RunwayConfigEnv", action: Action) -> float:
        return -env.cost(action) if env.reward_func == "linear" else 0.0

    def check_state(self, env: "RunwayConfigEnv") -> None:
        """Dynamically initialize a newly visited state and its feasible actions."""
        state = env.get_state()
        if state in self.states:
            return

        self.states[state] = True
        self.actions[state] = []
        self.q_values[state] = []
        self.N_a[state] = []

        arrival_limit = min(int(env.u[env.wc][-1]), int(env.arr))
        departure_envelope = env.v[env.wc]

        for arrivals_served in range(arrival_limit):
            departures_served = int(departure_envelope[arrivals_served])
            if env.dep >= departures_served:
                action = [arrivals_served, departures_served]
                self.actions[state].append(action)
                self.q_values[state].append(self._initial_q_value(env, action))
                self.N_a[state].append(1)

        departures_at_limit = int(departure_envelope[arrival_limit])
        if env.dep >= departures_at_limit:
            action = [arrival_limit, departures_at_limit]
        else:
            action = [arrival_limit, int(env.dep)]

        self.actions[state].append(action)
        self.q_values[state].append(self._initial_q_value(env, action))
        self.N_a[state].append(1)

    def _select_action_index(self, state: State, episode: int) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.choice(len(self.actions[state])))

        q = np.asarray(self.q_values[state], dtype=float)
        if self.ucb == 0:
            return int(np.argmax(q))

        counts = np.asarray(self.N_a[state], dtype=float)
        bonus = self.ucb * np.sqrt(np.log(episode + 1) / counts)
        return int(np.argmax(q + bonus))

    def take_epsilon_greedy_action(
        self, env: "RunwayConfigEnv", episode: int
    ) -> List[object]:
        state = env.get_state()
        action_index = self._select_action_index(state, episode)
        action = self.actions[state][action_index]

        self.N_a[state][action_index] += 1
        reward = env.react(action)
        self.G += reward
        self.cost += env.cost(action)
        return [action_index, action, reward]

    def greedy_action(self, state: State, action_index: int) -> int:
        greedy_index = int(np.argmax(self.q_values[state]))
        if np.isclose(
            self.q_values[state][greedy_index],
            self.q_values[state][action_index],
            atol=1e-6,
            rtol=0.0,
        ):
            return action_index
        return greedy_index

    def get_return(self) -> List[float]:
        return [self.cost, self.G]

    def compute_avg_return(self, weight: float, episode: int) -> List[float]:
        """Update running average cost and return.

        ``weight=1`` reproduces the arithmetic average used for Figure 4 of the
        paper. Values in ``(0,1)`` preserve the weighted-average option from
        the original implementation.
        """
        if not 0 < weight <= 1:
            raise ValueError("weight must be in (0, 1]")

        previous_cost = self.avg_cost
        previous_return = self.avg_G

        if weight == 1:
            self.avg_G = (self.avg_G * episode + self.G) / (episode + 1)
            self.avg_cost = (self.avg_cost * episode + self.cost) / (episode + 1)
        else:
            normalizer = (1 - weight) / (1 - weight ** (episode + 1))
            previous_mass = weight * (1 - weight**episode) / (1 - weight)
            self.avg_G = (self.avg_G * previous_mass + self.G) * normalizer
            self.avg_cost = (self.avg_cost * previous_mass + self.cost) * normalizer

        return [
            self.avg_cost,
            self.avg_G,
            self.avg_cost - previous_cost,
            self.avg_G - previous_return,
        ]

    def update_epsilon(self, episode: int) -> None:
        if episode >= self.decaying_steps:
            return

        if self.epsilon_decay == "exponential":
            self.epsilon *= (self.final_epsilon / self.initial_epsilon) ** (
                1 / self.decaying_steps
            )
        else:
            self.epsilon -= (
                self.initial_epsilon - self.final_epsilon
            ) / self.decaying_steps

        self.epsilon = max(self.final_epsilon, self.epsilon)

    def new_episode(self, episode: int) -> None:
        self.update_epsilon(episode)
        self.cost = 0.0
        self.G = 0.0


class SarsaAgent(_BaseAgent):
    """One-step SARSA agent retained for comparison with the paper method."""

    def policy_eval(
        self,
        curr_R: float,
        curr_state: State,
        curr_i_ac: int,
        state: Optional[State],
        i_ac: Optional[int],
    ) -> None:
        if state is None:
            delta = curr_R - self.q_values[curr_state][curr_i_ac]
        else:
            if i_ac is None:
                raise ValueError("i_ac is required for nonterminal SARSA updates")
            delta = (
                curr_R
                + self.discount_rate * self.q_values[state][i_ac]
                - self.q_values[curr_state][curr_i_ac]
            )
        self.q_values[curr_state][curr_i_ac] += self.step_size * delta


class QLearningAgent(_BaseAgent):
    """One-step Q-learning agent retained for comparison with Q(lambda)."""

    def policy_eval(
        self,
        curr_R: float,
        curr_state: State,
        curr_i_ac: int,
        state: Optional[State],
        i_ac: Optional[int],
    ) -> None:
        if state is None:
            delta = curr_R - self.q_values[curr_state][curr_i_ac]
        else:
            if i_ac is None:
                raise ValueError("i_ac is required for nonterminal Q-learning updates")
            greedy_index = self.greedy_action(state, i_ac)
            delta = (
                curr_R
                + self.discount_rate * self.q_values[state][greedy_index]
                - self.q_values[curr_state][curr_i_ac]
            )
        self.q_values[curr_state][curr_i_ac] += self.step_size * delta


class EligibilityTraceAgent(QLearningAgent):
    """Q(lambda) agent with eligibility traces used in the SITRAER paper."""

    def __init__(
        self,
        states: Optional[MutableMapping[State, bool]] = None,
        actions: Optional[MutableMapping[State, List[Action]]] = None,
        state_actions: Optional[MutableMapping[StateAction, List[int]]] = None,
        q_values: Optional[MutableMapping[State, List[float]]] = None,
        N_a: Optional[MutableMapping[State, List[int]]] = None,
        e: Optional[MutableMapping[StateAction, float]] = None,
        episode: int = 0,
        step_size: float = 0.5,
        discount_rate: float = 0.9,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 1e-6,
        decaying_steps: int = 100000,
        et_decay: float = 0.5,
        epsilon_decay: str = "exponential",
        ucb: float = 0.1,
    ) -> None:
        super().__init__(
            states=states,
            actions=actions,
            q_values=q_values,
            N_a=N_a,
            episode=episode,
            step_size=step_size,
            discount_rate=discount_rate,
            initial_epsilon=initial_epsilon,
            final_epsilon=final_epsilon,
            decaying_steps=decaying_steps,
            epsilon_decay=epsilon_decay,
            ucb=ucb,
        )
        if not 0 <= et_decay <= 1:
            raise ValueError("et_decay must be in [0, 1]")
        self.et_decay = float(et_decay)
        self.state_actions = {} if state_actions is None else state_actions
        self.e = {} if e is None else e

    def take_epsilon_greedy_action(
        self, env: "RunwayConfigEnv", episode: int
    ) -> List[object]:
        action_index, action, reward = super().take_epsilon_greedy_action(env, episode)
        state = env.get_state()
        key: StateAction = state + (int(action_index),)

        if key not in self.state_actions:
            self.state_actions[key] = list(state) + list(action)
        self.e.setdefault(key, 0.0)
        return [action_index, action, reward]

    def policy_eval(
        self,
        curr_R: float,
        curr_state: State,
        curr_i_ac: int,
        state: Optional[State],
        i_ac: Optional[int],
    ) -> None:
        current_key: StateAction = curr_state + (curr_i_ac,)
        self.e.setdefault(current_key, 0.0)

        if state is None:
            delta = curr_R - self.q_values[curr_state][curr_i_ac]
            next_action_is_greedy = False
        else:
            if i_ac is None:
                raise ValueError("i_ac is required for nonterminal Q(lambda) updates")
            greedy_index = self.greedy_action(state, i_ac)
            delta = (
                curr_R
                + self.discount_rate * self.q_values[state][greedy_index]
                - self.q_values[curr_state][curr_i_ac]
            )
            next_action_is_greedy = i_ac == greedy_index

        self.e[current_key] = (1 - self.step_size) * self.e[current_key] + 1

        for key in tuple(self.e):
            trace = self.e[key]
            if trace != 0:
                visited_state = key[:-1]
                visited_action_index = key[-1]
                self.q_values[visited_state][visited_action_index] += (
                    self.step_size * delta * trace
                )

            if state is not None:
                if next_action_is_greedy:
                    self.e[key] *= self.discount_rate * self.et_decay
                else:
                    self.e[key] = 0.0

    def new_episode(self, episode: int) -> None:
        self.e.clear()
        self.state_actions.clear()
        super().new_episode(episode)


class RunwayConfigEnv:
    """Environment for a fixed runway configuration and weather condition."""

    def __init__(
        self,
        weather_condition: int = 0,
        arrivals: int = 0,
        departures: int = 0,
        t: int = 0,
        arrival_priority: float = 2,
        u: Optional[MutableMapping[int, Sequence[int]]] = None,
        v: Optional[MutableMapping[int, Sequence[float]]] = None,
        c_1: float = 1e5,
        c_2: float = 1,
        reward_func: str = "linear",
    ) -> None:
        if arrival_priority <= 0:
            raise ValueError("arrival_priority must be positive")
        if c_2 <= 0:
            raise ValueError("c_2 must be positive")
        if reward_func not in {"linear", "hyperbolic", "exponential"}:
            raise ValueError("reward_func must be 'linear', 'hyperbolic', or 'exponential'")

        self.wc = int(weather_condition)
        self.arr = int(arrivals)
        self.dep = int(departures)
        self.t = int(t)
        self.arr_priority = float(arrival_priority)
        self.u = {} if u is None else u
        self.v = {} if v is None else v
        self.c_1 = float(c_1)
        self.c_2 = float(c_2)
        self.reward_func = reward_func

    def set_state(
        self, weather_condition: int, arrivals: int, departures: int, t: int
    ) -> None:
        if arrivals < 0 or departures < 0:
            raise ValueError("queue lengths cannot be negative")
        self.wc = int(weather_condition)
        self.arr = int(arrivals)
        self.dep = int(departures)
        self.t = int(t)

    def react(self, action: Sequence[int]) -> float:
        cost = self.cost(action)
        if self.reward_func == "hyperbolic":
            return self.c_1 / (self.c_2 + cost)
        if self.reward_func == "exponential":
            return self.c_1 * np.exp(-cost / self.c_2)
        return -cost

    def cost(self, action: Sequence[int]) -> float:
        return self.arr_priority * (self.arr - action[0]) ** 2 + (
            self.dep - action[1]
        ) ** 2

    def get_state(self) -> State:
        return (self.wc, self.arr, self.dep, self.t)


Sarsa_Agent = SarsaAgent
QL_Agent = QLearningAgent
ET_Agent = EligibilityTraceAgent
Runway_Config_Env = RunwayConfigEnv


__all__ = [
    "f",
    "capacity_envelope_departures",
    "SarsaAgent",
    "QLearningAgent",
    "EligibilityTraceAgent",
    "RunwayConfigEnv",
    "Sarsa_Agent",
    "QL_Agent",
    "ET_Agent",
    "Runway_Config_Env",
]
