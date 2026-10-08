# Runway Capacity Optimization with Reinforcement Learning

This repository contains the implementation associated with the conference paper:

**Lucas Orbolato Carvalho, Mayara Condé Rocha Murça, and Marcelo Xavier Guterres, "A Reinforcement Learning Approach for Optimizing the Runway Capacity Utilization Under Uncertainty," XX SITRAER - Air Transportation Symposium, Joinville, Brazil, 2023.**

Conference record: https://conferencias.ufsc.br/index.php/sitraer/xx/paper/view/2181

Full-text record: https://www.researchgate.net/publication/376480738_A_REINFORCEMENT_LEARNING_APPROACH_FOR_OPTIMIZING_THE_RUNWAY_CAPACITY_UTILIZATION_UNDER_UNCERTAINTY

> **Important:** This repository corresponds to the 2023 SITRAER conference study. It is **not** the implementation used in the later 2025 *Transportation Research Part A* paper, "A stochastic model-free reinforcement learning framework for optimizing runway capacity management under uncertainty."

## Code overview

The file `capoptimizer.py` contains the functions and classes used in the method. Different learning algorithms were implemented, including Monte Carlo, Sarsa, Q-learning, and Q(lambda). Q(lambda) was used in the study and generalizes the other implemented variants. Q-learning was selected instead of Sarsa because of its faster convergence in the experiments.

The file `general_cap_optimization.py` runs the reinforcement-learning algorithm using the classes and methods defined in `capoptimizer.py` and the data in `demand.xlsx`.

In `capacity_optimization_NLP_AMPL.py`, the problem is also solved with nonlinear programming using AMPL and Knitro for comparison.

## Research context

The study investigates runway-capacity utilization under uncertain weather conditions for a fixed runway configuration. The proposed approach uses tabular reinforcement learning with eligibility traces to learn capacity-allocation decisions without requiring an explicit stochastic model of the environment.
