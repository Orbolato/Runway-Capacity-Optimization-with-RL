# Runway-Capacity-Optimization-with-RL
Runway capacity optimization with Reinforcement Learning for a fixed runway configuration

The file 'capoptimizer.py' has the functions and classes used in the method. Different learning algorithms were implemented, like Monte Carlo, Sarsa, Q-learning and Q(lambda). The last was the one used in the study and generalizes all the others. Q-learning was chosen instead of Sarsa due to the accelerated convergence.

The file 'general_cap_optimization.py' runs the RL algorithm using the classes and methods defined in 'capoptimizer.py' and the data in 'demand.xlsx'.

In 'capacity_optimization_NLP_AMPL.py' the problem was also solved with non-linear programming using 'AMPL' with 'knitro'.
