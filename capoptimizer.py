import numpy as np

# Runway Configuration Capacity Envelope (RCCE)
def f(x,y,u): # Departures upper bound for the VMC curve
    v_ub = [] # Upper bound of departures v
    for i in range(len(u)): # For each arrival u_i
        for j in range(len(x)-1):  # For each intersecting point x_j
            if x[j] <= u[i] < x[j+1]: # If x_j <= u_i <= x_(j + 1)
                dy = y[j+1] - y[j]
                dx = x[j+1] - x[j]
                if dx == 0:
                    v_ub.append(y[j])
                else:
                    v_ub.append(dy/dx*(u[i] - x[j]) + y[j])
        if u[i] == x[-1]: # If u_i is the final point of the curve
            dy = y[-1] - y[-2]
            dx = x[-1] - x[-2]
            if dx == 0: # Vertical line
                v_ub.append(y[-2])
            else:
                v_ub.append(0)
        if u[i] > x[-1]: # If u_i is after the final point of the curve
            v_ub.append(0)
        if u[i] < 0: # If u_i is before the initial point of the curve
            v_ub.append(0)
    return v_ub

# Sarsa Agent
class Sarsa_Agent:
    
    def __init__(self, states = {}, actions = {}, q_values = {}, N_a = {},
                 episode = 0, step_size = 0.5, discount_rate = 0.9,
                 initial_epsilon = 1.0, final_epsilon = 1e-6,
                 decaying_steps = 100000, epsilon_decay = 'exponential',
                 ucb = 0.1):
        
        self.states = states # Set of states
        self.actions = actions # Set of actions
        self.q_values = q_values # Set of state-action pair values
        self.N_a = N_a # Counter for each action
        self.step_size = step_size # Step-size parameter
        self.discount_rate = discount_rate # Discount rate
        self.initial_epsilon = initial_epsilon # Initial epsilon
        self.epsilon = initial_epsilon # Current epsilon
        self.final_epsilon = final_epsilon # Final epsilon
        self.decaying_steps = decaying_steps
        self.epsilon_decay = epsilon_decay # Epsilon decay rate
        self.ucb = ucb # Upper confidence bound parameter
        
        # Current cost and return
        self.cost = 0 # Cost
        self.G = 0 # Return
        self.avg_cost = 0 # Average cost
        self.avg_G = 0 # Average return
    
    # Check if the state was already visited
    def check_state(self, env):
        
        state = env.get_state()
        
        if state not in self.states:
            
            # Initializing variables, parameters and lists
            self.states[state] = True # Visited state
            self.actions[state] = [] # Initializing actions
            self.q_values[state] = [] # Initializing state-action values
            self.N_a[state] = [] # Initializing action counter
            
            # Computing the boundaries of the actions set
            arr_lim = min(int(env.u[env.wc][-1]),env.arr)
            
            
            # Initializing the actions set and the action-value function
            for i in range(arr_lim):
                if env.dep >= int(env.v[env.wc][i]):
                    self.actions[state].append([i,int(env.v[env.wc][i])])
                    if env.reward_func == 'linear':
                        self.q_values[state].append(-(env.arr_priority*
                                                      (env.arr - i)**2 + 
                                                      (env.dep - 
                                                int(env.v[env.wc][i]))**2))
                    else:
                        self.q_values[state].append(0)
                    self.N_a[state].append(1)
                    
            if env.dep >= int(env.v[env.wc][arr_lim]):
                self.actions[state].append([arr_lim,
                                            int(env.v[env.wc][arr_lim])])
                if env.reward_func == 'linear':
                    self.q_values[state].append(-(env.arr_priority*
                                                  (env.arr - arr_lim)**2 + 
                                                  (env.dep - 
                                            int(env.v[env.wc][arr_lim]))**2))
                else:
                    self.q_values[state].append(0)
                self.N_a[state].append(1)
                
            else:
                self.actions[state].append([arr_lim,env.dep])
                if env.reward_func == 'linear':
                    self.q_values[state].append(-(env.arr_priority*
                                                  (env.arr - arr_lim)**2))
                else:
                    self.q_values[state].append(0)
                self.N_a[state].append(1)
        
    def take_epsilon_greedy_action(self, env, episode):
                
        # Setting the current state
        state = env.get_state()
        
        # Epsilon-greedy policy derived from Q
        p = np.random.rand()
        if p < self.epsilon:
            i_ac = np.random.choice(range(len(self.actions[state])))
        else:
            if self.ucb != 0:
                i_ac = np.argmax([self.q_values[state][i] + 
                                  self.ucb*np.sqrt(np.log(episode + 1) / 
                                                   self.N_a[state][i]) 
                                  for i in range(len(self.q_values[state]))])
            else:
                i_ac = np.argmax(self.q_values[state])
        ac = self.actions[state][i_ac]
        
        # Counting the action
        self.N_a[state][i_ac] += 1
        
        # Getting the environment reaction
        R = env.react(ac)
        self.G += R
        self.cost += env.cost(ac)
        
        # Return action
        return [i_ac, ac, R]
    
    def policy_eval(self, curr_R, curr_state, curr_i_ac, state, i_ac):
        
        if state is not None:

            # Calculating the increment
            delta = (curr_R + self.discount_rate*self.q_values[state][i_ac] 
                                        - self.q_values[curr_state][curr_i_ac])
        
        else:
            
            # Calculating the increment
            delta = (curr_R - self.q_values[curr_state][curr_i_ac])
            
        # Policy evaluation for the current state
        self.q_values[curr_state][curr_i_ac] += self.step_size*delta
    
    def get_return(self):
        
        return [self.cost, self.G]
    
    def compute_avg_return(self, weight, episode):
        
        dc = self.avg_cost
        dG = self.avg_G
        
        if weight == 1:
            self.avg_G = (self.avg_G*episode + self.G)/(episode + 1)
            self.avg_cost = (self.avg_cost*episode + self.cost)/(episode + 1)
        
        elif weight < 1:
            self.avg_G = (self.avg_G*weight*(1 - weight**episode) / 
                                        (1 - weight) + self.G)*((1 - weight) / 
                                                  (1 - weight**(episode + 1)))
            
            self.avg_cost = (self.avg_cost*weight*(1 - weight**episode) / 
                                     (1 - weight) + self.cost)*((1 - weight) / 
                                                  (1 - weight**(episode + 1)))
            
        else:
            print('Invalid weight for the average return.')
        
        dc = self.avg_cost - dc
        dG = self.avg_G - dG
        
        return [self.avg_cost, self.avg_G, dc, dG]
    
    def update_epsilon(self, episode): # Epsilon decay
        
        if episode < self.decaying_steps:
    
            if self.epsilon_decay == 'exponential':
                self.epsilon  *= (self.final_epsilon / 
                                  self.initial_epsilon)**(1 / 
                                                          self.decaying_steps)
            
            elif self.epsilon_decay == 'linear':
                self.epsilon -= (self.initial_epsilon - 
                                 self.final_epsilon) / self.decaying_steps
            
            else:
                print('Invalid decaying epsilon strategy, keeping it constant.')
    
    def new_episode(self, episode):
        self.update_epsilon(episode)
        self.cost = 0
        self.G = 0

# Q-Learning Agent
class QL_Agent:
    
    def __init__(self, states = {}, actions = {}, q_values = {}, N_a = {},
                 episode = 0, step_size = 0.5, discount_rate = 0.9,
                 initial_epsilon = 1.0, final_epsilon = 1e-6,
                 decaying_steps = 100000, epsilon_decay = 'exponential',
                 ucb = 0.1):
        
        self.states = states # Set of states
        self.actions = actions # Set of actions
        self.q_values = q_values # Set of state-action pair values
        self.N_a = N_a # Counter for each action
        self.step_size = step_size # Step-size parameter
        self.discount_rate = discount_rate # Discount rate
        self.initial_epsilon = initial_epsilon # Initial epsilon
        self.epsilon = initial_epsilon # Current epsilon
        self.final_epsilon = final_epsilon # Final epsilon
        self.decaying_steps = decaying_steps
        self.epsilon_decay = epsilon_decay # Epsilon decay rate
        self.ucb = ucb # Upper confidence bound parameter
        
        # Current cost and return
        self.cost = 0 # Cost
        self.G = 0 # Return
        self.avg_cost = 0 # Average cost
        self.avg_G = 0 # Average return
    
    # Check if the state was already visited
    def check_state(self, env):
        
        state = env.get_state()
        
        if state not in self.states:
            
            # Initializing variables, parameters and lists
            self.states[state] = True # Visited state
            self.actions[state] = [] # Initializing actions
            self.q_values[state] = [] # Initializing state-action values
            self.N_a[state] = [] # Initializing action counter
            
            # Computing the boundaries of the actions set
            arr_lim = min(int(env.u[env.wc][-1]),env.arr)
            
            
            # Initializing the actions set and the action-value function
            for i in range(arr_lim):
                if env.dep >= int(env.v[env.wc][i]):
                    self.actions[state].append([i,int(env.v[env.wc][i])])
                    if env.reward_func == 'linear':
                        self.q_values[state].append(-(env.arr_priority*
                                                      (env.arr - i)**2 + 
                                                      (env.dep - 
                                                int(env.v[env.wc][i]))**2))
                    else:
                        self.q_values[state].append(0)
                    self.N_a[state].append(1)
                    
            if env.dep >= int(env.v[env.wc][arr_lim]):
                self.actions[state].append([arr_lim,
                                            int(env.v[env.wc][arr_lim])])
                if env.reward_func == 'linear':
                    self.q_values[state].append(-(env.arr_priority*
                                                  (env.arr - arr_lim)**2 + 
                                                  (env.dep - 
                                            int(env.v[env.wc][arr_lim]))**2))
                else:
                    self.q_values[state].append(0)
                self.N_a[state].append(1)
                
            else:
                self.actions[state].append([arr_lim,env.dep])
                if env.reward_func == 'linear':
                    self.q_values[state].append(-(env.arr_priority*
                                                  (env.arr - arr_lim)**2))
                else:
                    self.q_values[state].append(0)
                self.N_a[state].append(1)
        
    def take_epsilon_greedy_action(self, env, episode):
                
        # Setting the current state
        state = env.get_state()
        
        # Epsilon-greedy policy derived from Q
        p = np.random.rand()
        
        if p < self.epsilon:
            i_ac = np.random.choice(range(len(self.actions[state])))
        
        else:
            if self.ucb != 0:
                i_ac = np.argmax([self.q_values[state][i] + 
                                  self.ucb*np.sqrt(np.log(episode + 1) / 
                                                   self.N_a[state][i]) 
                                  for i in range(len(self.q_values[state]))])
            
            else:
                i_ac = np.argmax(self.q_values[state])
        
        ac = self.actions[state][i_ac]
        
        # Counting the action
        self.N_a[state][i_ac] += 1
        
        # Getting the environment reaction
        R = env.react(ac)
        self.G += R
        self.cost += env.cost(ac)
        
        # Return action
        return [i_ac, ac, R]
    
    def greedy_action(self, state, i_ac):
        
        # Greedy action
        i_ac_opt = np.argmax(self.q_values[state])
        
        # Checking if it is optimal
        if np.abs(self.q_values[state][i_ac_opt] - 
                  self.q_values[state][i_ac]) < 1e-6:
            i_ac_opt = i_ac
        
        # Returning greedy action
        return i_ac_opt
    
    def policy_eval(self, curr_R, curr_state, curr_i_ac, state, i_ac):
        
        if state is not None:
            
            # Greedy action
            i_ac_opt = self.greedy_action(state, i_ac)
            
            # Calculating the increment
            delta = (curr_R + self.discount_rate*self.q_values[state][i_ac_opt] 
                                        - self.q_values[curr_state][curr_i_ac])
        
        else:
            
            # Calculating the increment
            delta = (curr_R - self.q_values[curr_state][curr_i_ac])
            
        # Policy evaluation for the current state
        self.q_values[curr_state][curr_i_ac] += self.step_size*delta
    
    def get_return(self):
        
        return [self.cost, self.G]
    
    def compute_avg_return(self, weight, episode):
        
        dc = self.avg_cost
        dG = self.avg_G
        
        if weight == 1:
            self.avg_G = (self.avg_G*episode + self.G)/(episode + 1)
            self.avg_cost = (self.avg_cost*episode + self.cost)/(episode + 1)
        
        elif weight < 1:
            self.avg_G = (self.avg_G*weight*(1 - weight**episode) / 
                                        (1 - weight) + self.G)*((1 - weight) / 
                                                  (1 - weight**(episode + 1)))
            
            self.avg_cost = (self.avg_cost*weight*(1 - weight**episode) / 
                                     (1 - weight) + self.cost)*((1 - weight) / 
                                                  (1 - weight**(episode + 1)))
            
        else:
            print('Invalid weight for the average return.')
        
        dc = self.avg_cost - dc
        dG = self.avg_G - dG
        
        return [self.avg_cost, self.avg_G, dc, dG]
    
    def update_epsilon(self, episode): # Epsilon decay
        
        if episode < self.decaying_steps:
    
            if self.epsilon_decay == 'exponential':
                self.epsilon  *= (self.final_epsilon / 
                                  self.initial_epsilon)**(1 / 
                                                          self.decaying_steps)
            
            elif self.epsilon_decay == 'linear':
                self.epsilon -= (self.initial_epsilon - 
                                 self.final_epsilon) / self.decaying_steps
            
            else:
                print('Invalid decaying epsilon strategy, keeping it constant.')
    
    def new_episode(self, episode):
        self.update_epsilon(episode)
        self.cost = 0
        self.G = 0

# Eligibility Trace Agent
class ET_Agent:
    
    def __init__(self, states = {}, actions = {}, state_actions = {},
                 q_values = {}, N_a = {}, e = {}, episode = 0, step_size = 0.5,
                 discount_rate = 0.9, initial_epsilon = 1.0,
                 final_epsilon = 1e-6, decaying_steps = 100000, et_decay = 0.5,
                 epsilon_decay = 'exponential', ucb = 0.1):
        
        self.states = states # Set of states
        self.actions = actions # Set of actions
        self.state_actions = state_actions
        self.q_values = q_values # Set of state-action pair values
        self.N_a = N_a # Counter for each action
        self.e = e # Eligibility traces
        self.step_size = step_size # Step-size parameter
        self.discount_rate = discount_rate # Discount rate
        self.initial_epsilon = initial_epsilon # Initial epsilon
        self.epsilon = initial_epsilon # Current epsilon
        self.final_epsilon = final_epsilon # Final epsilon
        self.decaying_steps = decaying_steps
        self.et_decay = et_decay # Eligibility trace decay rate
        self.epsilon_decay = epsilon_decay # Epsilon decay rate
        self.ucb = ucb # Upper confidence bound parameter
        
        # Current cost and return
        self.cost = 0 # Cost
        self.G = 0 # Return
        self.avg_cost = 0 # Average cost
        self.avg_G = 0 # Average return
    
    # Check if the state was already visited
    def check_state(self, env):
        
        state = env.get_state()
        
        if state not in self.states:
            
            # Initializing variables, parameters and lists
            self.states[state] = True
            self.actions[state] = [] # Initializing actions
            self.q_values[state] = [] # Initializing state-action values
            self.N_a[state] = []
            
            # Computing the boundaries of the actions set
            arr_lim = min(int(env.u[env.wc][-1]),env.arr)
            
            
            # Initializing the actions set and the action-value function
            for i in range(arr_lim):
                if env.dep >= int(env.v[env.wc][i]):
                    self.actions[state].append([i,int(env.v[env.wc][i])])
                    if env.reward_func == 'linear':
                        self.q_values[state].append(-(env.arr_priority*
                                                      (env.arr - i)**2 + 
                                                      (env.dep - 
                                                int(env.v[env.wc][i]))**2))
                    else:
                        self.q_values[state].append(0)
                    self.N_a[state].append(1)
                    
            if env.dep >= int(env.v[env.wc][arr_lim]):
                self.actions[state].append([arr_lim,
                                            int(env.v[env.wc][arr_lim])])
                if env.reward_func == 'linear':
                    self.q_values[state].append(-(env.arr_priority*
                                                  (env.arr - arr_lim)**2 + 
                                                  (env.dep - 
                                            int(env.v[env.wc][arr_lim]))**2))
                else:
                    self.q_values[state].append(0)
                self.N_a[state].append(1)
                
            else:
                self.actions[state].append([arr_lim,env.dep])
                if env.reward_func == 'linear':
                    self.q_values[state].append(-(env.arr_priority*
                                                  (env.arr - arr_lim)**2))
                else:
                    self.q_values[state].append(0)
                self.N_a[state].append(1)
        
    def take_epsilon_greedy_action(self, env, episode):
                
        # Setting the current state
        state = env.get_state()
        
        # Epsilon-greedy policy derived from Q
        p = np.random.rand()
        
        if p < self.epsilon:
            i_ac = np.random.choice(range(len(self.actions[state])))
        
        else:
            if self.ucb != 0:
                i_ac = np.argmax([self.q_values[state][i] + 
                                  self.ucb*np.sqrt(np.log(episode + 1) / 
                                                   self.N_a[state][i]) 
                                  for i in range(len(self.q_values[state]))])
            else:
                i_ac = np.argmax(self.q_values[state])
        
        ac = self.actions[state][i_ac]
        
        # Counting the action
        self.N_a[state][i_ac] += 1
        
        # Getting the environment reaction
        R = env.react(ac)
        self.G += R
        self.cost += env.cost(ac)
        
        # Keeping visited state-action
        if state + (i_ac,) not in self.state_actions:
            self.state_actions[state + (i_ac,)] = list(state) + ac    
        
        # Initializing eligibility trace
        if state + (i_ac,) not in self.e:
            self.e[state + (i_ac,)] = 0
        
        # Return action
        return [i_ac, ac, R]
    
    def greedy_action(self, state, i_ac):
        
        # Greedy action
        i_ac_opt = np.argmax(self.q_values[state])
        
        # Checking if it is optimal
        if np.abs(self.q_values[state][i_ac_opt] - 
                  self.q_values[state][i_ac]) < 1e-6:
            i_ac_opt = i_ac
        
        # Returning greedy action
        return i_ac_opt
    
    def policy_eval(self, curr_R, curr_state, curr_i_ac, state, i_ac):
        
        if state is not None:
            
            # Greedy action
            i_ac_opt = self.greedy_action(state, i_ac)
            
            # Calculating the increment
            delta = (curr_R + self.discount_rate*self.q_values[state][i_ac_opt] 
                                        - self.q_values[curr_state][curr_i_ac])
            
            # Updating eligibility trace
            self.e[curr_state + (curr_i_ac,)] *= (1 - self.step_size)
            self.e[curr_state + (curr_i_ac,)] += 1
            
            # Policy evaluation
            for sa in self.state_actions:
                vis_state = sa[0:4]
                vis_i_ac = sa[4]
                self.q_values[vis_state][vis_i_ac] += (self.step_size*delta*
                                            self.e[vis_state + (vis_i_ac,)])
                
                if i_ac == i_ac_opt:
                    self.e[vis_state + (vis_i_ac,)] *= (self.discount_rate*
                                                        self.et_decay)
                
                else:
                    self.e[vis_state + (vis_i_ac,)] = 0
        
        else:
            
            # Calculating the increment
            delta = (curr_R - self.q_values[curr_state][curr_i_ac])
            
            # Updating eligibility trace
            self.e[curr_state + (curr_i_ac,)] *= (1 - self.step_size)
            self.e[curr_state + (curr_i_ac,)] += 1
            
            # Policy evaluation
            for sa in self.state_actions:
                vis_state = sa[0:4]
                vis_i_ac = sa[4]
                self.q_values[vis_state][vis_i_ac] += (self.step_size*delta*
                                            self.e[vis_state + (vis_i_ac,)])
    
    def get_return(self):
        
        return [self.cost, self.G]
    
    def compute_avg_return(self, weight, episode):
        
        dc = self.avg_cost
        dG = self.avg_G
        
        if weight == 1:
            self.avg_G = (self.avg_G*episode + self.G)/(episode + 1)
            self.avg_cost = (self.avg_cost*episode + self.cost)/(episode + 1)
        
        elif weight < 1:
            self.avg_G = (self.avg_G*weight*(1 - weight**episode) / 
                                        (1 - weight) + self.G)*((1 - weight) / 
                                                  (1 - weight**(episode + 1)))
            
            self.avg_cost = (self.avg_cost*weight*(1 - weight**episode) / 
                                     (1 - weight) + self.cost)*((1 - weight) / 
                                                  (1 - weight**(episode + 1)))
            
        else:
            print('Invalid weight for the average return.')
        
        dc = self.avg_cost - dc
        dG = self.avg_G - dG
        
        return [self.avg_cost, self.avg_G, dc, dG]
    
    def update_epsilon(self, episode): # Epsilon decay
        
        if episode < self.decaying_steps:
    
            if self.epsilon_decay == 'exponential':
                self.epsilon  *= (self.final_epsilon / 
                                  self.initial_epsilon)**(1 / 
                                                          self.decaying_steps)
            
            elif self.epsilon_decay == 'linear':
                self.epsilon -= (self.initial_epsilon - 
                                 self.final_epsilon) / self.decaying_steps
            
            else:
                print('Invalid decaying epsilon strategy, keeping it constant.')
    
    def new_episode(self, episode):
        self.e = {} # Restart eligibility traces
        self.state_actions = {} # Clean visited state-action pairs
        self.update_epsilon(episode)
        self.cost = 0
        self.G = 0

# Environment for a given runway configuration
class Runway_Config_Env:
    
    def __init__(self, weather_condition = 0, arrivals = 0, departures = 0,
                 t = 0, arrival_priority = 2, u = [], v = [], c_1 = 1e5,
                 c_2 = 1, reward_func = 'linear'):
        self.wc = weather_condition # Weather condition
        self.arr = arrivals # Number of arrivals in queue
        self.dep = departures # Number of departures in queue
        self.t = t # Time slot/period
        self.arr_priority = arrival_priority # Arrival priority
        self.u = u # Lists of allowed arrivals for each weather condition
        self.v = v # Lists of maximum departures for each weather condition
        self.c_1 = c_1 # Constant for the reward function
        self.c_2 = c_2 # Constant for the reward function
        self.reward_func = reward_func # Reward function type
    
    # Set a state
    def set_state(self, weather_condition, arrivals, departures, t):
        self.wc = weather_condition # Weather condition
        self.arr = arrivals # Number of arrivals in queue
        self.dep = departures # Number of departures in queue
        self.t = t # Time slot/period
    
    # Return a reward
    def react(self, ac):
        
        if self.reward_func == 'hyperbolic':
            
            return self.c_1 / (self.c_2 + self.arr_priority*
                               (self.arr - ac[0])**2 + 
                               (self.dep - ac[1])**2)
        
        elif self.reward_func == 'exponential':
            
            return self.c_1*np.exp(-(self.arr_priority*(self.arr - ac[0])**2 + 
                               (self.dep - ac[1])**2) / self.c_2)
        
        elif self.reward_func == 'linear':
            
            return -(self.arr_priority*(self.arr - ac[0])**2 + 
                               (self.dep - ac[1])**2)
        else:
            print('Invalid reward function type.')
    
    # Return a cost
    def cost(self, ac):
        
        return self.arr_priority*(self.arr - ac[0])**2 + (self.dep - ac[1])**2
    
    # Return the environment state
    def get_state(self):
        
        return (self.wc, self.arr, self.dep, self.t)
    