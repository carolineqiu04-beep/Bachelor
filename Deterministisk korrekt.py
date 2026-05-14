import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, qmc
from gurobipy import *
import random
# -------------------------
# Data
# -------------------------
wind_data = pd.read_excel("/Users/carolineqiu/Desktop/Data/Forecasts_Hour (2).xlsx")

# Konverter tid
wind_data['HourDK'] = pd.to_datetime(wind_data['HourDK'])

wind_filtered = wind_data[
    (wind_data['ForecastType'].isin(["Offshore Wind", "Onshore Wind"])) &
    (wind_data['PriceArea'] == "DK2") &
    (~wind_data['HourDK'].dt.year.isin([2019, 2026])) &
    (wind_data['HourDK'].dt.month.isin([12,1,2]))   # vinter
].copy()

dayahead = (
    wind_filtered
    .dropna(subset=['ForecastDayAhead'])
    .groupby('HourDK', as_index=False)['ForecastDayAhead']
    .sum()
)

dayahead['hour'] = dayahead['HourDK'].dt.hour

avg_winter_day = (
    dayahead
    .groupby('hour')['ForecastDayAhead']
    .mean()
)

demand_data = pd.read_excel("/Users/carolineqiu/Desktop/Data/IEEE Demand.xlsx", header=None)
demand_matrix = demand_data.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
N = demand_matrix.shape[0]   # antal busser
T = demand_matrix.shape[1]

D = {n: {t: demand_matrix[n,t] for t in range(T)} for n in range(N)}
print("\n--- Samlet efterspørgsel per time ---")
for t in range(T):
    total_demand = sum(D[n][t] for n in range(N))
    print(f"t={t:02d}: {total_demand:.0f} MW")

ieee_data = pd.read_excel(
    "/Users/carolineqiu/Desktop/Data/IEEE data-kopi.xlsx",
    header=None,        # Første række er data, ikke header
    decimal=','         # Fortolk komma som decimal
)

ieee_data.columns = [
    "generator_location_bus_id",
    "P_underline", # Minimum produktionsniveau
    "P_bar", # Maksimal produktionsniveau
    "c", # Variable omkostninger
    "f", # Faste omkostninger
    "C^3", # Opstartsomkostning i kategori 3 (kold)
    "C^2", # Opstartsomkostning i kategori 2 (lunken)
    "C^1", # Opstartsomkostning i kategori 1 (varm)
    "UT_i", #Minimum driftstid (up time)
    "DT_i", #Minimum sluktid (down time)
    "RU", # Ramp up rate
    "RD", # Ramp down rate
    "SU", #Start up ramp limit
    "SD", #Shut down ramp limit
    "T^3", # Antal perioder enhed i skal være slukket for at være i kategori 3 (kold)
    "T^2", # Antal perioder enhed i skal være slukket for at være i kategori 2 (lunken)
    "T^1", # Antal perioder enhed i skal være slukket for at være i kategori 2 (varm)
    "DT0" # Initial down time
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


# -------------------------
# Model
# -------------------------
m = Model("DUC")

#------- Parametre --------
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
c_US = 50000   # undersupply penalty
c_OS = 50000    # oversupply penalty
S_i = [len(C[i]) for i in range(I)] # Mængden af kategorier
w = {t: avg_winter_day.loc[t] for t in range(24)}

#------- Variable --------
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
p = {}
s_plus = {}
s_minus = {}
for i in range(I):
    for t in range(T):
        p[i,t] = m.addVar(lb=0, name=f"p_{i}_{t}")
for n in range(N):
    for t in range(T):
        s_plus[n,t] = m.addVar(lb=0, name=f"splus_{n}_{t}")
        s_minus[n,t] = m.addVar(lb=0, name=f"sminus_{n}_{t}")
# (1.2)–(1.4)
for i in range(I):
    if DT0[i] > 0:
        m.addConstr(u[i,0] == 0)  
        m.addConstr(y[i,0] == 0)  
        m.addConstr(v[i,0] == 0)  
    else:
        m.addConstr(u[i,0] == 1)  
        m.addConstr(y[i,0] == 0)  
        m.addConstr(v[i,0] == 0)  
#(1.5)-(1.6)
for i in range(I):
    for t in range(1, T):
        m.addConstr(u[i,t] - u[i,t-1] == v[i,t] - y[i,t])
for i in range(I):
    for t in range(1, T):
        m.addConstr(v[i,t] + y[i,t] <= 1)
#(1.7)
for i in range(I):
    for t in range(T):
        for k in range(t, min(t + int(UT_i[i]), T)):
            m.addConstr(u[i,k] >= v[i,t])
#(1.8)
for i in range(I):
    if DT0[i] > 0: 
        for t in range(min(int(DT_i[i]), T)):
            m.addConstr(u[i, t] == 0)
        for t in range(int(DT_i[i]), T - 1):
            for k in range(t + 1, min(t + int(DT_i[i]), T)):
                m.addConstr(u[i, k] <= 1 - y[i, t])
        for t in range(T - 1):
            for k in range(t + 1, min(t + int(DT_i[i]), T)):
                m.addConstr(u[i, k] <= 1 - y[i, t])
#(1.9)
    for t in range(T):
        m.addConstr(v[i,t] == quicksum(delta[i,t,s] for s in range(S_i[i])))
#(1.10)-(1.12)
for i in range(I):
    for t in range(1, T):
        for s in range(S_i[i]):
            interval = T_s[i][s]
            start = min(interval)
            slut  = max(interval)
            relevant_y = quicksum(
                y[i, t - k] for k in range(start, slut + 1)
                if t - k >= 0
            )
            if t < start:
                if s == S_i[i] - 1:
                    m.addConstr(delta[i, t, s] <= v[i, t])
                else:
                    m.addConstr(delta[i, t, s] == 0)
            else:
                m.addConstr(delta[i, t, s] <= relevant_y)
#(1.14)-(1.15)
for i in range(I):
    for t in range(T):
        m.addConstr(p[i,t] <= P_bar[i] * u[i,t])  
        m.addConstr(P_underline[i] * u[i,t] <= p[i,t]) 
#(1.16)-(1.17)
for i in range(I):
    for t in range(T-1):
        m.addConstr(p[i,t] <= P_bar[i]*(u[i,t] - y[i,t+1]) + SD[i]*y[i,t+1])
        m.addConstr( p[i,t]  <= P_bar[i]*(u[i,t+1] - v[i,t]) + SU[i]*v[i,t])

#(1.18)-(1.19)
for i in range(I):
    for t in range(1,T):
        m.addConstr(p[i,t] - p[i,t-1]
                    <= (SU[i]-P_underline[i]-RU[i])*v[i,t]
                    + (P_underline[i]+RU[i])*u[i,t]
                    - P_underline[i]*u[i,t-1])
        m.addConstr(p[i,t-1] - p[i,t]
                    <= (SD[i]-P_underline[i]-RD[i])*y[i,t]
                    + (P_underline[i]+RD[i])*u[i,t-1]
                    - P_underline[i]*u[i,t])
#(1.20)
for t in range(T):
    m.addConstr(
        quicksum(p[i,t] for i in range(I))
        + w[t]
        + quicksum(s_plus[n,t] - s_minus[n,t] for n in range(N))
        ==
        quicksum(D[n][t] for n in range(N)))
    
#Objektfunktion (1.1)
m.setObjective(
    quicksum(f[i]*u[i,t] + c_SU[i,t] for i in range(I) for t in range(T))
    + quicksum(c[i]*p[i,t] for i in range(I) for t in range(T))
    + quicksum(c_US*s_minus[n,t] + c_OS*s_plus[n,t] for n in range(N) for t in range(T)),
    GRB.MINIMIZE)

#------- Løsning --------
m.optimize()
print('Objective value: %g' % m.objVal)


