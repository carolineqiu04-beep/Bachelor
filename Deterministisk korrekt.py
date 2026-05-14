import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, qmc
from gurobipy import *
import random

SEED = 42

# Sæt seeds
random.seed(SEED)
np.random.seed(SEED)

m = Model("DUC")
m.setParam('Seed', SEED)
m.setParam('Threads', 1) 
m.setParam('Method', 2)

# -------------------------
# 1. Læs data
# -------------------------
wind_data = pd.read_excel("/Users/carolineqiu/Desktop/Data/Forecasts_Hour (2).xlsx")

# Konverter tid
wind_data['HourDK'] = pd.to_datetime(wind_data['HourDK'])

# -------------------------
# 2. Filtrer relevante data, så det kun er vinter sæson
# -------------------------
wind_filtered = wind_data[
    (wind_data['ForecastType'].isin(["Offshore Wind", "Onshore Wind"])) &
    (wind_data['PriceArea'] == "DK2") &
    (~wind_data['HourDK'].dt.year.isin([2019, 2026])) &
    (wind_data['HourDK'].dt.month.isin([12,1,2]))   # vinter
].copy()

# -------------------------
# 3. Day-ahead forecast
# -------------------------
dayahead = (
    wind_filtered
    .dropna(subset=['ForecastDayAhead'])
    .groupby('HourDK', as_index=False)['ForecastDayAhead']
    .sum()
)

# Udtræk time på dagen
dayahead['hour'] = dayahead['HourDK'].dt.hour

# Gennemsnit for hver time
avg_winter_day = (
    dayahead
    .groupby('hour')['ForecastDayAhead']
    .mean()
)

print(avg_winter_day)

# -------------------------
# 4. Plot af gennemsnitlig vinterdag
# -------------------------
#plt.figure(figsize=(8,5))

#plt.plot(avg_winter_day.index, avg_winter_day.values, marker='o')

#plt.xlabel("Time på dagen")
#plt.ylabel("Gennemsnitlig vindforecast (MW)")
#plt.title("Gennemsnitlig vinterdag – vindproduktion DK2")
#plt.xticks(range(24))
#plt.grid(True)

#plt.show()
# 1. Læs data
# -------------------------
demand_data = pd.read_excel("/Users/carolineqiu/Desktop/Data/IEEE Demand.xlsx", header=None)
demand_matrix = demand_data.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
N = demand_matrix.shape[0]   # antal busser
T = demand_matrix.shape[1]
# Summér over alle busser for hver periode (axis=0)
D = {n: {t: demand_matrix[n,t] for t in range(T)} for n in range(N)}
print("\n--- Samlet efterspørgsel per time ---")
for t in range(T):
    total_demand = sum(D[n][t] for n in range(N))
    print(f"t={t:02d}: {total_demand:.0f} MW")

# Læs Excel uden at bruge første række som header, og angiv decimalseparator
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

#------- Parametre --------
I = ieee_data.shape[0]
#------- Parametre --------

C = {}
for i in range(I):
    C[i] = [
        ieee_data.loc[i,"C^1"],
        ieee_data.loc[i,"C^2"],
        ieee_data.loc[i,"C^3"]
    ]

T_s = {}
for i in range(I):
    dt = int(DT_i[i])
    warm     = range(dt,      dt + 3)      # varm:   slukket DT_i til DT_i+2 timer
    lukewarm = range(dt + 3,  dt + 6)      # lunken: slukket DT_i+3 til DT_i+5 timer
    cold     = range(dt + 6,  dt + 16)     # kold:   slukket DT_i+6+ timer
    T_s[i] = [warm, lukewarm, cold]


c_US = 50000   # undersupply penalty
c_OS = 50000    # oversupply penalty

S_i = [len(C[i]) for i in range(I)] # Mængden af kategorier

w = {t: avg_winter_day.loc[t] for t in range(24)}




#------- First stage decision variables --------
u = m.addVars(I, T, vtype=GRB.BINARY, name="u")     # on/off
v = m.addVars(I, T, vtype=GRB.BINARY, name="v")     # start
y = m.addVars(I, T, vtype=GRB.BINARY, name="y")     # shutdown
delta = {}
for i in range(I):
    for t in range(T):
        for s in range(S_i[i]):
            delta[i,t,s] = m.addVar(vtype=GRB.BINARY, name=f"delta_{i}_{t}_{s}")
# c_SU beregnes fra delta
c_SU = {}
for i in range(I):
    for t in range(T):
        c_SU[i,t] = quicksum(C[i][s] * delta[i,t,s] for s in range(S_i[i]))

#------- First stage decision variables som tidligere var second--------
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

#------- First-stage Bi-betingelser --------
#------- First-stage Bi-betingelser --------
# Initial status baseret på DT0
for i in range(I):
    if DT0[i] > 0:
        m.addConstr(u[i,0] == 0)
        m.addConstr(y[i,0] == 0)
        m.addConstr(v[i,0] == 0)
    else:
        m.addConstr(u[i,0] == 1)
        m.addConstr(y[i,0] == 0)
        m.addConstr(v[i,0] == 0)

# Minimum sluktid - her skal vi også tage højde for initial sluktid
for i in range(I):
    if DT0[i] > 0:  # Hvis enheden starter slukket
        # Den skal forblive slukket i de første DT_i perioder
        for t in range(min(int(DT_i[i]), T)):
            m.addConstr(u[i, t] == 0)

        # Derefter kan den tidligst starte efter DT_i perioder
        # og skal overholde minimum sluktid
        for t in range(int(DT_i[i]), T - 1):
            for k in range(t + 1, min(t + int(DT_i[i]), T)):
                m.addConstr(u[i, k] <= 1 - y[i, t])

    # For generatorer der starter tændt
    else:
        for t in range(T - 1):
            for k in range(t + 1, min(t + int(DT_i[i]), T)):
                m.addConstr(u[i, k] <= 1 - y[i, t])

# Logisk sammenhæng mellem u, v og y for t >= 1
for i in range(I):
    for t in range(1, T):
        m.addConstr(u[i,t] - u[i,t-1] == v[i,t] - y[i,t])

# v og y kan ikke begge være 1 i samme periode
for i in range(I):
    for t in range(1, T):
        m.addConstr(v[i,t] + y[i,t] <= 1)

# Tilføj dette efter "u[i,t] - u[i,t-1] == v[i,t] - y[i,t]" blokken
'''
# En enhed kan ikke starte og stoppe i samme periode
for i in range(I):
    for t in range(T):
        m.addConstr(v[i,t] + y[i,t] <= 1)

# Hvis en enhed starter, skal den være tændt
for i in range(I):
    for t in range(T):
        m.addConstr(v[i,t] <= u[i,t])

# Hvis en enhed stopper, skal den være slukket i næste periode
for i in range(I):
    for t in range(T-1):
        m.addConstr(y[i,t] <= 1 - u[i,t+1])

# Minimum driftstid
for i in range(I):
    for t in range(T-1):
        for k in range(t+1, min(t + int(UT_i[i]), T)):
            m.addConstr(u[i,k] >= v[i,t])

# Minimum sluktid
for i in range(I):
    for t in range(T-1):
        for k in range(t+1, min(t + int(DT_i[i]), T)):
            m.addConstr(u[i,k] <= 1 - y[i,t])
'''
# Minimum driftstid (inkluderer start-perioden)
for i in range(I):
    for t in range(T):
        for k in range(t, min(t + int(UT_i[i]), T)):
            m.addConstr(u[i,k] >= v[i,t])

# Minimum sluktid (inkluderer stop-perioden)
for i in range(I):
    for t in range(T):
        for k in range(t, min(t + int(DT_i[i]), T)):
            m.addConstr(u[i,k] <= 1 - y[i,t])

# Opstartsindikator
# Opstartsindikator baseret på shutdown i specifikt tidsinterval
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
                # Kold opstart fra begyndelsen
                if s == S_i[i] - 1:
                    m.addConstr(delta[i, t, s] <= v[i, t])
                else:
                    m.addConstr(delta[i, t, s] == 0)
            else:
                m.addConstr(delta[i, t, s] <= relevant_y)

# Sammenkobling af opstart og kategori
for i in range(I):
    for t in range(T):
        m.addConstr(v[i,t] == quicksum(delta[i,t,s] for s in range(S_i[i])))

#------- First-stage Bi-betingelser som tidligere var second stage--------
#Max og min produktion
for i in range(I):
    for t in range(T):
        m.addConstr(p[i,t] <= P_bar[i] * u[i,t])  # max produktion
        m.addConstr(P_underline[i] * u[i,t] <= p[i,t]) # min produktion

# Nedsluknings- og opstartshastighed
for i in range(I):
    for t in range(T-1):
        m.addConstr(p[i,t] <= P_bar[i]*(u[i,t] - y[i,t+1]) + SD[i]*y[i,t+1])
        m.addConstr( p[i,t]  <= P_bar[i]*(u[i,t+1] - v[i,t]) + SU[i]*v[i,t])

# Ramp up og Ramp down
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

# Samlet efterspørgsel
for t in range(T):
    m.addConstr(
        quicksum(p[i,t] for i in range(I))
        + w[t]
        + quicksum(s_plus[n,t] - s_minus[n,t] for n in range(N))
        ==
        quicksum(D[n][t] for n in range(N))
    )

#------- Objektfunktion--------
m.setObjective(
    quicksum(f[i]*u[i,t] + c_SU[i,t] for i in range(I) for t in range(T))
    + quicksum(c[i]*p[i,t] for i in range(I) for t in range(T))
    + quicksum(c_US*s_minus[n,t] + c_OS*s_plus[n,t] for n in range(N) for t in range(T)),
    GRB.MINIMIZE
)

#------- Løsning --------
m.optimize()
print('Objective value: %g' % m.objVal)


