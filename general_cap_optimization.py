# Loading useful libraries/packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import capoptimizer as cp

# Plotting the VMC capacity envelope
print('Plotting the VMC  and the IMC capacity envelopes...')
x_vmc = [0, 17, 24, 28, 28] # Intersecting points in x
y_vmc = [30, 30, 24, 15, 0] # Intersecting points in y
plt.plot(x_vmc, y_vmc)

# Plotting the IMC capacity envelope
x_imc = [0, 12, 17, 20, 20] # Intersecting points in x
y_imc = [22, 22, 17, 11, 0] # Intersecting points in y
x = {0: x_vmc, 1: x_imc}
y = {0: y_vmc, 1: y_imc}
plt.plot(x_imc, y_imc)
plt.legend(['VMC', 'IMC'])
plt.xlabel('Arrivals')
plt.ylabel('Departures')
plt.show()
print('Done.\n')

# Loading data
data_file = (input('Enter the dataset exact file name with its extension'
                   + ' (e.g.: demand.xlsx): ') or 'demand.xlsx')
print('Loading data...')
demand_data = pd.read_excel(data_file, index_col = 0, decimal = ',')
print('Data loaded.\n')

# Correcting the indexes to avoid errors
print('Correcting the indexes to avoid errors...')
initial_indexes = demand_data.index
demand_data.index = ['Slot ' + str(i + 1) for i in range(len(initial_indexes))]
print('Done.\n')

# Total number of time slots
N = len(demand_data)

# Defining the arrival priority
alpha = float(input('Enter the arrival priority (float number, default = 2): ')
              or '2')

# Starting queues
a_0 = int(input('Enter the initial arrival queue length'
                + ' (integer >= 0, default = 0): ') or '0')
d_0 = int(input('Enter the initial departure queue length'
                + ' (integer >= 0, default = 0): ') or '0')

# Maximum length of the queues
a_max = np.sum(demand_data['Arrival Demand'])
d_max = np.sum(demand_data['Departure Demand'])

# Actions for each state (the actions set depend only on the weather condition)
u_vmc = [i for i in range(np.max(x_vmc)+1)]
v_vmc = cp.f(x_vmc, y_vmc, u_vmc)
u_imc = [i for i in range(np.max(x_imc)+1)]
v_imc = cp.f(x_imc, y_imc, u_imc)
u = {0: u_vmc,1: u_imc}
v = {0: v_vmc,1: v_imc}

# List of average returns
list_avg_G = []

# List of average costs
list_avg_c = []

# Number of episodes
episodes = 2000000

# Step-size parameter for nonstationary and incremental implementation
# (between 0 and 1)
step_size = 0.9

# Discount rate for TD algorithms (between 0 and 1)
discount_rate = 0.9

# Eligibility trace decay-rate parameter (between 0 and 1)
et_decay = 0.5

# Epsilon values for decaying epsilon-greedy policy (real)
initial_epsilon = 1
final_epsilon = 1e-16
decaying_steps = episodes / 2

# Reward function scale
scale = 1e3

# Constant c_1 of the reward function (don't change)
c_1 = scale*(alpha*np.dot(demand_data['Arrival Demand'], 
                          demand_data['Arrival Demand'])
                     + np.dot(demand_data['Departure Demand'], 
                              demand_data['Departure Demand']))

# Constant c_2 of the reward function
c_2 = scale

# Upper confidence bound
ucb = 0 # 0.1*c_1 / c_2

# Weight for the weighted average of the returns and costs (0 < w <= 1)
w = 1 # (1e-12)**(1 / episodes)

# Creating environment
env = cp.Runway_Config_Env(arrival_priority = alpha, u = u, v = v,
                           c_1 = c_1, c_2 = c_2, reward_func = 'linear')

# Creating agent
agent = cp.ET_Agent(step_size = step_size, discount_rate = discount_rate,
                    initial_epsilon = initial_epsilon,
                    final_epsilon = final_epsilon,
                    decaying_steps = decaying_steps, et_decay = et_decay,
                    epsilon_decay = 'exponential', ucb = ucb)

# agent = cp.QL_Agent(step_size = step_size, discount_rate = discount_rate,
#                     initial_epsilon = initial_epsilon,
#                     final_epsilon = final_epsilon,
#                     decaying_steps = decaying_steps,
#                     epsilon_decay = 'exponential', ucb = ucb)

# State-action value iteration
for ep in range(episodes):
    
    # Queues in the beginning of the episode
    a_t = a_0
    d_t = d_0
    
    # Starting time
    t = 0
    
    # Updating the demand queue
    a_t += demand_data['Arrival Demand'][t]
    d_t += demand_data['Departure Demand'][t]
    
    # Probability of IMC
    IMC_prob = demand_data['IMC Probability'][t]

    # Determining the current weather condition
    wc = np.random.choice([0, 1], p = [1 - IMC_prob, IMC_prob])
    
    # Setting the environment state
    env.set_state(wc, a_t, d_t, t)
    
    # Checking if the environemnt state was already visited
    agent.check_state(env)

    # Take an epsilon-greedy action over the environment
    [i_ac, ac, R] = agent.take_epsilon_greedy_action(env, ep)
    
    # Updating t
    t += 1
    
    # For each time period
    while t < N:
           
        # Saving the current state
        curr_state = env.get_state()
        curr_i_ac = i_ac
        curr_R = R
        
        # Updating the demand queue
        a_t += demand_data['Arrival Demand'][t] - ac[0]
        d_t += demand_data['Departure Demand'][t] - ac[1]
        
        # Probability of IMC
        IMC_prob = demand_data['IMC Probability'][t]
    
        # Determining the current weather condition
        wc = np.random.choice([0, 1], p = [1 - IMC_prob, IMC_prob])
        
        # Setting the environment state
        env.set_state(wc, a_t, d_t, t)
        
        # Checking if the environment state was already visited
        agent.check_state(env)

        # Take an epsilon-greedy action over the environment
        [i_ac, ac, R] = agent.take_epsilon_greedy_action(env, ep)
        
        # Policy evaluation
        agent.policy_eval(curr_R, curr_state, curr_i_ac, env.get_state(), i_ac)
        
        # Updating t
        t += 1
    
    # Policy evaluation
    agent.policy_eval(R, env.get_state(), i_ac, None, None)
    
    # Getting the return and cost
    [c, G] = agent.get_return()
    
    # Calculating the average return and cost
    [avg_c, avg_G, dc, dG] = agent.compute_avg_return(w, ep)
    list_avg_G.append(avg_G)
    list_avg_c.append(avg_c)
    print('Episode ' + str(ep+1)
          + ': c = '
          + str(c)
          + ', G_m = ' 
          + str(round(avg_G,2))
          + ', c_m = ' 
          + str(round(avg_c,2)))
    
    # End of episode
    agent.new_episode(ep)

print('Plotting the learning curve...')
plt.plot(range(1,episodes+1),list_avg_c)
plt.xlabel('Episodes')
plt.ylabel('Delay/Cost')
plt.show()
plt.plot(range(1,episodes+1),list_avg_G)
plt.xlabel('Episodes')
plt.ylabel('Return')
plt.show()
print('Done.')
