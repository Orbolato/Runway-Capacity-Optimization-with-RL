# Loading useful libraries/packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from amplpy import AMPL,Environment

# Plotting the VMC capacity envelope
print('Plotting the VMC  and the IMC capacity envelopes...')
x_vmc = [0,17,24,28,28] # Intersecting points in x
y_vmc = [30,30,24,15,0] # Intersecting points in y
plt.plot(x_vmc,y_vmc)

# Plotting the IMC capacity envelope
x_imc = [0,12,17,20,20] # Intersecting points in x
y_imc = [22,22,17,11,0] # Intersecting points in y
plt.plot(x_imc,y_imc)
plt.legend(['VMC','IMC'])
plt.xlabel('Arrivals')
plt.ylabel('Departures')
plt.show()
print('Done.\n')

# Loading data
print('Loading data...')
demand_data = pd.read_excel("demand.xlsx", index_col = 0) # Demand information
print('Done.')

# Total number of time slots
N = len(demand_data)

# Arrival priority
alpha = 2

# Initial queue
a_0 = 0 # Initial arrival queue
d_0 = 0 # Initial departure queue

# Optimal values list
opt_values = []

# Optimization for different scenarios
episodes = 1000

for eps in range(episodes):
    
    a_t = a_0
    d_t = d_0
    z = 0
    arr = {}
    dep = {}
    plt.plot(x_vmc,y_vmc)
    plt.plot(x_imc,y_imc)
    plt.legend(['VMC','IMC'])
    plt.xlabel('Arrivals')
    plt.ylabel('Departures')
    
    for t in range(N):
        
        # Updating the demand queue
        a_t += demand_data['Arrival Demand'][t]
        d_t += demand_data['Departure Demand'][t]
        
        # Probability of IMC
        IMC_prob = demand_data['IMC Probability'][t]
    
        # Determining the current state
        wc = np.random.choice(['IMC','VMC'],p = [IMC_prob,1 - IMC_prob])
        
        # Determining the constraints
        p = []
        q = []
        r = []
        if wc == 'VMC':
            p.append(1)
            q.append(0)
            r.append(np.max(y_vmc))
            for i in range(1,len(x_vmc)-2):
                dy = y_vmc[i+1] - y_vmc[i]
                dx = x_vmc[i+1] - x_vmc[i]
                p.append(1)
                q.append(-dy/dx)
                r.append(y_vmc[i] - dy/dx*x_vmc[i])
            p.append(0)
            q.append(1)
            r.append(np.max(x_vmc))
        elif wc == 'IMC':
            p.append(1)
            q.append(0)
            r.append(np.max(y_imc))
            for i in range(1,len(x_imc)-2):
                dy = y_imc[i+1] - y_imc[i]
                dx = x_imc[i+1] - x_imc[i]
                p.append(1)
                q.append(-dy/dx)
                r.append(y_imc[i] - dy/dx*x_imc[i])
            p.append(0)
            q.append(1)
            r.append(np.max(x_imc))
        
        # Solving with AMPL
        ampl = AMPL(Environment(r'C:\Users\lucas\Desktop\ampl.mswin64'))
        ampl.eval(r"""
            param a_t;
            param d_t;
            param alpha;
            param p{i in 1..4};
            param q{i in 1..4};
            param r{i in 1..4};
            var u integer >= 0;
            var v integer >= 0;
            minimize delay:
                alpha*(a_t - u)^2 + (d_t - v)^2;
            s.t. c1:
                p[1]*v + q[1]*u <= r[1];
            s.t. c2:
                p[2]*v + q[2]*u <= r[2];
            s.t. c3:
                p[3]*v + q[3]*u <= r[3];
            s.t. c4:
                p[4]*v + q[4]*u <= r[4];
            s.t. c5:
                u <= a_t;
            s.t. c6:
                v <= d_t;
        """)
        ampl.param['alpha'] = alpha
        ampl.param['a_t'] = int(a_t)
        ampl.param['d_t'] = int(d_t)
        ampl.param['p'] = p
        ampl.param['q'] = q
        ampl.param['r'] = r
        ampl.option['solver'] = 'knitro'
        ampl.option['knitro_options'] = 'ms_enable=1 ms_num_to_save=5 ms_savetol=0.01 ms_maxsolves=10'
        ampl.solve()
        #assert ampl.get_value('solve_result') == 'solved'
        sol = ampl.get_value('alpha*(a_t - u)^2 + (d_t - v)^2')
        arr[t] = round(ampl.get_value('u'))
        dep[t] = round(ampl.get_value('v'))
        print('Optimal solution for episode '+str(eps+1)+', step '+str(t+1)+': '+str(round(sol,2))+', x = ['+str(arr[t])+', '+str(dep[t])+']')
        plt.plot(arr[t],dep[t],'o',color='grey')
        
        # Updating the queues
        a_t -= arr[t]
        d_t -= dep[t]
        
        # Calculating the total solution
        z += sol
    
    opt_values.append(round(z,2))
    print('Optimal value for episode '+str(eps+1)+': '+str(round(z,2)))
    plt.show()

# Plotting the results
plt.plot([i+1 for i in range(episodes)],opt_values,'o')
plt.plot([i+1 for i in range(episodes)],[round(np.average(opt_values),2) for i in range(episodes)],'--')
plt.xlabel('Episodes')
plt.ylabel('Delay/Cost')
plt.show()
print('Optimal value (average): '+str(round(np.average(opt_values),2)))
        