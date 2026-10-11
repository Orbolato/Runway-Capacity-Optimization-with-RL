import numpy as np

import capoptimizer as cp


X_VMC = [0, 17, 24, 28, 28]
Y_VMC = [30, 30, 24, 15, 0]
X_IMC = [0, 12, 17, 20, 20]
Y_IMC = [22, 22, 17, 11, 0]


def envelopes():
    u_vmc = list(range(29))
    u_imc = list(range(21))
    return (
        {0: u_vmc, 1: u_imc},
        {
            0: cp.f(X_VMC, Y_VMC, u_vmc),
            1: cp.f(X_IMC, Y_IMC, u_imc),
        },
    )


def test_capacity_envelope_matches_paper_breakpoints():
    result = cp.f(X_VMC, Y_VMC, [0, 17, 24, 27, 28, 29, -1])
    assert np.allclose(result, [30, 30, 24, 17.25, 15, 0, 0])


def test_dynamic_action_set_respects_capacity_and_queue_bounds():
    u, v = envelopes()
    env = cp.RunwayConfigEnv(
        weather_condition=0,
        arrivals=3,
        departures=10,
        t=0,
        arrival_priority=2,
        u=u,
        v=v,
    )
    agent = cp.EligibilityTraceAgent(ucb=0)
    agent.check_state(env)

    state = env.get_state()
    assert agent.actions[state] == [[3, 10]]
    assert agent.q_values[state] == [-0.0]


def test_linear_reward_is_negative_paper_cost():
    env = cp.RunwayConfigEnv(arrivals=12, departures=9, arrival_priority=2)
    action = [8, 7]
    expected_cost = 2 * (12 - 8) ** 2 + (9 - 7) ** 2
    assert env.cost(action) == expected_cost
    assert env.react(action) == -expected_cost


def test_agent_default_storage_is_not_shared_between_instances():
    a = cp.EligibilityTraceAgent()
    b = cp.EligibilityTraceAgent()
    a.states[(0, 1, 2, 3)] = True
    assert (0, 1, 2, 3) not in b.states


def test_terminal_q_lambda_update_matches_algorithm_one():
    u, v = envelopes()
    env = cp.RunwayConfigEnv(
        weather_condition=0,
        arrivals=1,
        departures=1,
        t=0,
        arrival_priority=2,
        u=u,
        v=v,
    )
    agent = cp.EligibilityTraceAgent(
        step_size=0.5,
        discount_rate=0.9,
        et_decay=0.5,
        initial_epsilon=0,
        final_epsilon=0,
        epsilon_decay="linear",
        ucb=0,
    )
    agent.check_state(env)
    i_ac, _, reward = agent.take_epsilon_greedy_action(env, 0)
    state = env.get_state()
    q_before = agent.q_values[state][i_ac]

    expected_delta = reward - q_before
    expected_q = q_before + 0.5 * expected_delta
    agent.policy_eval(reward, state, i_ac, None, None)

    assert np.isclose(agent.q_values[state][i_ac], expected_q)


def test_exponential_epsilon_reaches_requested_endpoint():
    agent = cp.EligibilityTraceAgent(
        initial_epsilon=1.0,
        final_epsilon=0.01,
        decaying_steps=2,
        epsilon_decay="exponential",
    )
    agent.new_episode(0)
    assert np.isclose(agent.epsilon, 0.1)
    agent.new_episode(1)
    assert np.isclose(agent.epsilon, 0.01)


def test_backward_compatible_class_aliases():
    assert cp.ET_Agent is cp.EligibilityTraceAgent
    assert cp.QL_Agent is cp.QLearningAgent
    assert cp.Sarsa_Agent is cp.SarsaAgent
    assert cp.Runway_Config_Env is cp.RunwayConfigEnv
