import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from scipy.stats import weibull_min, qmc

# Data

demand_data = pd.read_excel("/Users/carolineqiu/Desktop/Data/IEEE Demand.xlsx", header=None)
demand_matrix = demand_data.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
N, T = demand_matrix.shape
D = {n: {t: demand_matrix[n, t] for t in range(T)} for n in range(N)}

ieee_data = pd.read_excel(
    "/Users/carolineqiu/Desktop/Data/IEEE data-kopi.xlsx",
    header=None, decimal=','
)
ieee_data.columns = ["generator_location_bus_id", "P_underline", "P_bar", "c", "f", "C^3", "C^2",
                     "C^1", "UT_i", "DT_i", "RU", "RD", "SU", "SD", "T^3", "T^2", "T^1", "DT0" 
]

P_bar = ieee_data["P_bar"].values
P_underline = ieee_data["P_underline"].values
UT_i = ieee_data["UT_i"].values
DT_i = ieee_data["DT_i"].values
RU = ieee_data["RU"].values
RD = ieee_data["RD"].values
SU = ieee_data["SU"].values
SD = ieee_data["SD"].values
c = ieee_data["c"].values
f = ieee_data["f"].values
T1 = ieee_data["T^1"].values
T2 = ieee_data["T^2"].values
T3 = ieee_data["T^3"].values
DT0 = ieee_data["DT0"].values
I = ieee_data.shape[0]

# Scenarier
wind_data = pd.read_excel("/Users/carolineqiu/Desktop/Data/Forecasts_Hour (2).xlsx")
wind_data['HourDK'] = pd.to_datetime(wind_data['HourDK'])

wind_filtered = wind_data[
    wind_data['ForecastType'].isin(["Offshore Wind", "Onshore Wind"]) &
    (wind_data['PriceArea'] == "DK2") &
    ~wind_data['HourDK'].dt.year.isin([2019, 2026]) &
    wind_data['HourDK'].dt.month.isin([12, 1, 2])
].copy()

dayahead = (wind_filtered.dropna(subset=['ForecastDayAhead'])
            .groupby('HourDK', as_index=False)['ForecastDayAhead'].sum())
current  = (wind_filtered.dropna(subset=['ForecastCurrent'])
            .groupby('HourDK', as_index=False)['ForecastCurrent'].sum())

wind = pd.merge(dayahead, current, on='HourDK', how='inner').set_index('HourDK')

noise           = wind['ForecastCurrent'] - wind['ForecastDayAhead']
abs_noise       = noise.abs()
positive_share  = (noise > 0).mean()
negative_share  = 1.0 - positive_share

daily_noise_profile = noise.groupby(noise.index.hour).mean().values 

hourly_weibull_params = []
for h in range(24):
    hour_noise = abs_noise[noise.index.hour == h] 
    shape_h, loc_h, scale_h = weibull_min.fit(hour_noise, floc=0) 
    hourly_weibull_params.append((shape_h, scale_h)) 

n_hours = 24
n_scenarios = 200
rho = 0.6
scenarios = range(n_scenarios)

sampler = qmc.LatinHypercube(d=n_hours, seed=SEED)
lhs_sample = sampler.random(n=n_scenarios) 

noise_sim_lhs = np.zeros((n_hours, n_scenarios))

for h in range(n_hours):
    shape_h, scale_h = hourly_weibull_params[h]
    u = lhs_sample[:, h]  
    noise_size = weibull_min.ppf(u, shape_h, scale=scale_h)
    sign = np.random.choice([-1,1], size=n_scenarios, p=[negative_share, positive_share])
    noise_sim_lhs[h, :] = noise_size * sign

for s in range(n_scenarios):
    prev = 0
    for h in range(n_hours):
        noise_sim_lhs[h, s] = rho * prev + noise_sim_lhs[h, s]
        prev = noise_sim_lhs[h, s]

dayahead_day = wind['ForecastDayAhead'].head(24).values.reshape(-1,1)
simulated_lhs_current = np.maximum(dayahead_day + noise_sim_lhs,0)

p_scen_prob = {xi: 1/n_scenarios for xi in scenarios}

w = {}
for xi in scenarios:
    for t in range(T):
        w[t,xi] = max(0, simulated_lhs_current[t,xi])

# Model
m = Model("SUC")

# Parametre
C = {}
for i in range(I):
    C[i] = [
        ieee_data.loc[i,"C^1"],
        ieee_data.loc[i,"C^2"],
        ieee_data.loc[i,"C^3"]]
T_s = {}
for i in range(I):
    dt = int(DT_i[i])
    warm     = range(dt,      dt + 3)      
    lukewarm = range(dt + 3,  dt + 6)      
    cold     = range(dt + 6,  dt + 16)     
    T_s[i] = [warm, lukewarm, cold]
c_US = 50000  
c_OS = 50000 
S_i = [len(C[i]) for i in range(I)] 

# Første-fase beslutningsvariable

u = m.addVars(I, T, vtype=GRB.BINARY, name="u")  
v = m.addVars(I, T, vtype=GRB.BINARY, name="v")   
y = m.addVars(I, T, vtype=GRB.BINARY, name="y") 
delta = {}
for i in range(I):
    for t in range(T):
        for s in range(S_i[i]):
            delta[i,t,s] = m.addVar(vtype=GRB.BINARY, name=f"delta_{i}_{t}_{s}")
c_SU = {}
for i in range(I):
    for t in range(T):
        c_SU[i,t] = quicksum(C[i][s] * delta[i,t,s] for s in range(S_i[i]))
        
# Anden-fase beslutningsvariable
p = {}
s_plus = {}
s_minus = {}
for xi in scenarios:
    for i in range(I):
        for t in range(T):
            p[i,t,xi] = m.addVar(lb=0, name=f"p_{i}_{t}_{xi}")
    for n in range(N):
        for t in range(T):
            s_plus[n,t,xi] = m.addVar(lb=0, name=f"splus_{n}_{t}_{xi}")
            s_minus[n,t,xi] = m.addVar(lb=0, name=f"sminus_{n}_{t}_{xi}")
            
# Bi-betinglinger hørende til første-fase

# (2.2)-(2.4)
for i in range(I):
    if DT0[i] > 0:
        m.addConstr(u[i,0] == 0)
        m.addConstr(y[i,0] == 0)
        m.addConstr(v[i,0] == 0)
    else:
        m.addConstr(u[i,0] == 1)
        m.addConstr(y[i,0] == 0)
        m.addConstr(v[i,0] == 0)
# (2.5)
for i in range(I):
    for t in range(1, T):
        m.addConstr(u[i,t] - u[i,t-1] == v[i,t] - y[i,t])

 # (2.6)
for i in range(I):
    for t in range(1, T):
        m.addConstr(v[i,t] + y[i,t] <= 1)
# (2.7)
for i in range(I):
    for t in range(T):
        for k in range(t, min(t + int(UT_i[i]), T)):
            m.addConstr(u[i,k] >= v[i,t])
# (2.8)
for i in range(I):
    if DT0[i] > 0: 
        for t in range(min(int(DT_i[i]), T)):
            m.addConstr(u[i, t] == 0)
        for t in range(int(DT_i[i]), T - 1):
            for k in range(t + 1, min(t + int(DT_i[i]), T)):
                m.addConstr(u[i, k] <= 1 - y[i, t])
    else:
        for t in range(T - 1):
            for k in range(t + 1, min(t + int(DT_i[i]), T)):
                m.addConstr(u[i, k] <= 1 - y[i, t])
# (2.9)
for i in range(I):
    for t in range(T):
        m.addConstr(v[i,t] == quicksum(delta[i,t,s] for s in range(S_i[i])))
# (2.10)-(2.12)
for i in range(I):
    for t in range(1, T):
        for s in range(S_i[i]):
            interval = T_s[i][s]
            start = min(interval)
            slut  = max(interval)
            relevant_y = quicksum(
                y[i, t - k] for k in range(start, slut + 1)
                if t - k >= 0)
            if t < start:
                if s == S_i[i] - 1:
                    m.addConstr(delta[i, t, s] <= v[i, t])
                else:
                    m.addConstr(delta[i, t, s] == 0)
            else:
                m.addConstr(delta[i, t, s] <= relevant_y)

# Bi-betinglinger hørende til anden-fase

# (2.16)-(2.17)
for xi in scenarios:
    for i in range(I):
        for t in range(T):
            m.addConstr(p[i,t,xi]  <= P_bar[i] * u[i,t]) 
            m.addConstr(P_underline[i] * u[i,t] <= p[i,t,xi]) 

# (2.18)-(2.19)
for xi in scenarios:
    for i in range(I):
        for t in range(T-1):
            m.addConstr(p[i,t,xi] <= P_bar[i]*(u[i,t]-y[i,t+1]) + SD[i]*y[i,t+1])
            m.addConstr(p[i,t,xi]  <= P_bar[i]*(u[i,t+1]-v[i,t]) + SU[i]*v[i,t])

# (2.20)-(2.21)
for xi in scenarios:
    for i in range(I):
        for t in range(1,T):
            m.addConstr(p[i,t,xi] - p[i,t-1,xi]  <= (SU[i]-P_underline[i]-RU[i])*v[i,t] + (P_underline[i]+RU[i])*u[i,t] - P_underline[i]*u[i,t-1])
            m.addConstr(p[i,t-1,xi] - p[i,t,xi] <= (SD[i]-P_underline[i]-RD[i])*y[i,t] + (P_underline[i]+RD[i])*u[i,t-1] - P_underline[i]*u[i,t])
# (2.22)
for xi in scenarios:
    for t in range(T):
        m.addConstr(
            quicksum(p[i,t,xi] for i in range(I)) +
            w[t,xi] +
            quicksum(s_plus[n,t,xi] - s_minus[n,t,xi] for n in range(N))  ==
            quicksum(D[n][t] for n in range(N)))

# Objektfunktion

first_stage_cost = quicksum(f[i]*u[i,t] + c_SU[i,t] for i in range(I) for t in range(T))
# (2.15)
expected_Q = 0
for xi in scenarios:
    Q_xi = quicksum(
        c[i]*p[i,t,xi] for i in range(I) for t in range(T)
    ) + quicksum(
        c_US*s_minus[n,t,xi] + c_OS*s_plus[n,t,xi] for n in range(N) for t in range(T)
    )
    expected_Q += p_scen_prob[xi]*Q_xi
# (2.1)
m.setObjective(first_stage_cost + expected_Q, GRB.MINIMIZE)

# Løsning
m.optimize()
print('Objective value: %g' % m.objVal)
