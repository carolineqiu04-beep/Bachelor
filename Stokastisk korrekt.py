import pandas as pd
from gurobipy import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, qmc
import random
import json
import webbrowser
import os

# Sæt seed for reproducerbare resultater
SEED = 42  # eller et andet tal

# Sæt seed for numpy
np.random.seed(SEED)

# Sæt seed for pythons random-modul (bruges af nogle funktioner)
random.seed(SEED)

# Opret model
m = Model("SUC")
m.setParam('Seed', SEED)
m.setParam('Threads', 1)        # Reproducerbarhed
m.setParam('Method', 2)

# 1. Læs data
# -------------------------
demand_data = pd.read_excel("/Users/carolineqiu/Desktop/Data/IEEE Demand.xlsx", header=None)
demand_matrix = demand_data.apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
N = demand_matrix.shape[0]   # antal busser
T = demand_matrix.shape[1]
# Summér over alle busser for hver periode (axis=0)
D = {n: {t: demand_matrix[n,t] for t in range(T)} for n in range(N)}

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
# -------------------------

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


#------- Scenarier --------
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

# -------------------------
# 4. Current forecast
# -------------------------
current = (
    wind_filtered
    .dropna(subset=['ForecastCurrent'])
    .groupby('HourDK', as_index=False)['ForecastCurrent']
    .sum()
)

# -------------------------
# 5. Merge forecasts
# -------------------------
wind = pd.merge(dayahead, current, on='HourDK', how='inner')

## Sæt index
wind.set_index('HourDK', inplace=True)

# Beregn støj
noise = wind['ForecastCurrent'] - wind['ForecastDayAhead']
abs_noise = np.abs(noise)

# Positive / negative andel
positive_share = (noise > 0).mean()
negative_share = (noise < 0).mean()
print(f"Andel positiv støj: {positive_share:.3f}")
print(f"Andel negativ støj: {negative_share:.3f}")

# Gennemsnitlig fejl pr. time
daily_noise_profile = noise.groupby(noise.index.hour).mean().values  # 24 timer
print(daily_noise_profile)

# Weibull pr. time
hourly_weibull_params = []
for h in range(24):
    hour_noise = abs_noise[noise.index.hour == h]  # Absolut fejl for time h
    shape_h, loc_h, scale_h = weibull_min.fit(hour_noise, floc=0) # Fit Weibull
    hourly_weibull_params.append((shape_h, scale_h))  # Gem shape og scale

#for h, (shape_h, scale_h) in enumerate(hourly_weibull_params):
    #print(f"Time {h:02d}: shape (k) = {shape_h:.4f}, scale (c) = {scale_h:.2f}")


# -------------------------
# Latin (Nok denne du skal bruge)
# -------------------------
n_hours = 24
n_scenarios = 200
rho = 0.6
scenarios = range(n_scenarios)

# positive/negative andel
positive_share = 0.505
negative_share = 0.495

# LHS sampling
sampler = qmc.LatinHypercube(d=n_hours, seed=SEED)  # dimension = 24 timer
lhs_sample = sampler.random(n=n_scenarios)  # n_scenarios = antal kolonner

noise_sim_lhs = np.zeros((n_hours, n_scenarios))

for h in range(n_hours):
    shape_h, scale_h = hourly_weibull_params[h]
    # Brug LHS til at få 'noise_size' for time h
    u = lhs_sample[:, h]  # LHS-probabilities for time h
    noise_size = weibull_min.ppf(u, shape_h, scale=scale_h)  # Invers CDF
    # Tilføj fortegn baseret på empiriske sandsynligheder
    sign = np.random.choice([-1,1], size=n_scenarios, p=[negative_share, positive_share])
    noise_sim_lhs[h, :] = noise_size * sign

# Autokorrelation (kan tilføjes hvis ønsket)
for s in range(n_scenarios):
    prev = 0
    for h in range(n_hours):
        noise_sim_lhs[h, s] = rho * prev + noise_sim_lhs[h, s]
        prev = noise_sim_lhs[h, s]

dayahead_day = wind['ForecastDayAhead'].head(24).values.reshape(-1,1)
simulated_lhs_current = np.maximum(dayahead_day + noise_sim_lhs,0)

n_scenarios = simulated_lhs_current.shape[1]

scenarios = range(n_scenarios)

p_scen_prob = {xi: 1/n_scenarios for xi in scenarios}

# Vind parameter
w = {}
for xi in scenarios:
    for t in range(T):
        w[t,xi] = max(0, simulated_lhs_current[t,xi])

# Print støjled og simulerede vindscenarier
for xi in scenarios:
    print(f"\nScenarie {xi}:")
    for t in range(T):
        print(f"  t={t:02d}: ε={noise_sim_lhs[t,xi]:7.1f} MW, W={w[t,xi]:7.1f} MW")

print("\n--- Vind per time (gennemsnit) ---")
for t in range(T):
    avg = np.mean([w[t, xi] for xi in scenarios])
    print(f"t={t:02d}: {avg:.1f} MW")


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

#------- Second stage decision variables --------
# Opret variabler pr. scenarie
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

#------- Second-stage Bi-betingelser --------
#Max og min produktion
for xi in scenarios:
    for i in range(I):
        for t in range(T):
            m.addConstr(p[i,t,xi]  <= P_bar[i] * u[i,t])         # max produktion
            m.addConstr(P_underline[i] * u[i,t] <= p[i,t,xi])               # min produktion

# Nedsluknings- og opstartshastighed
for xi in scenarios:
    for i in range(I):
        for t in range(T-1):  # undgå t+1 > T
            m.addConstr(p[i,t,xi] <= P_bar[i]*(u[i,t]-y[i,t+1]) + SD[i]*y[i,t+1])
            m.addConstr(p[i,t,xi]  <= P_bar[i]*(u[i,t+1]-v[i,t]) + SU[i]*v[i,t])

# Ramp up og Ramp down
for xi in scenarios:
    for i in range(I):
        for t in range(1,T):
            # Ramp-up
            m.addConstr(p[i,t,xi] - p[i,t-1,xi]  <= (SU[i]-P_underline[i]-RU[i])*v[i,t] + (P_underline[i]+RU[i])*u[i,t] - P_underline[i]*u[i,t-1])
            # Ramp-down
            m.addConstr(p[i,t-1,xi] - p[i,t,xi] <= (SD[i]-P_underline[i]-RD[i])*y[i,t] + (P_underline[i]+RD[i])*u[i,t-1] - P_underline[i]*u[i,t])

# Samlet efterspørgsel
for xi in scenarios:
    for t in range(T):
        m.addConstr(
            quicksum(p[i,t,xi] for i in range(I)) +
            w[t,xi] +
            quicksum(s_plus[n,t,xi] - s_minus[n,t,xi] for n in range(N))  ==
            quicksum(D[n][t] for n in range(N))
        )


#------- Objektfunktion--------
first_stage_cost = quicksum(f[i]*u[i,t] + c_SU[i,t] for i in range(I) for t in range(T))

expected_Q = 0
for xi in scenarios:
    Q_xi = quicksum(
        c[i]*p[i,t,xi] for i in range(I) for t in range(T)
    ) + quicksum(
        c_US*s_minus[n,t,xi] + c_OS*s_plus[n,t,xi] for n in range(N) for t in range(T)
    )
    expected_Q += p_scen_prob[xi]*Q_xi

m.setObjective(first_stage_cost + expected_Q, GRB.MINIMIZE)

#------- Løsning --------
m.optimize()
print('Objective value: %g' % m.objVal)

#------- Kul Løsning --------
coal_idx = ieee_data[ieee_data["P_bar"] >= 300].index.tolist()

print("\n=== KUL-TURBINER LØSNING ===")
print(f"Kul-generatorer: {coal_idx}")
print()

# On/off status
print("--- On/Off status (u) ---")
header = "Gen  |" + "".join(f" t{t:02d}" for t in range(T))
print(header)
for i in coal_idx:
    row = f"{i:4d} |"
    for t in range(T):
        row += f"  {int(round(u[i,t].X))}"
    print(row)

# Tænd status (v)
print("\n--- Tænd status (v) ---")
print(header)
for i in coal_idx:
    row = f"{i:4d} |"
    for t in range(T):
        val = int(round(v[i,t].X))
        row += f"  {'↑' if val==1 else '-'}"
    print(row)

# Sluk status (y)
print("\n--- Sluk status (y) ---")
print(header)
for i in coal_idx:
    row = f"{i:4d} |"
    for t in range(T):
        val = int(round(y[i,t].X))
        row += f"  {'↓' if val==1 else '-'}"
    print(row)

# Produktion
print("\n--- Produktion p (MW) ---")
print(header)
for i in coal_idx:
    row = f"{i:4d} |"
    for t in range(T):
        avg_prod = np.mean([p[i, t, xi].X for xi in scenarios])
        row += f" {avg_prod:5.0f}"
    print(row)

# Opsummering
print("\n--- Opsummering ---")
for i in coal_idx:
    timer_on = sum(int(round(u[i,t].X)) for t in range(T))
    total_p = sum(np.mean([p[i, t, xi].X for xi in scenarios]) for t in range(T))
    print(f"Gen {i}: P_bar={P_bar[i]} MW | Tændt {timer_on}/24 timer | Total produktion: {total_p:.0f} MWh")

#------- Gas + damp Løsning --------
gas_idx = [i for i in range(I) if i not in coal_idx]

print("\n=== GAS/DAMP-TURBINER LØSNING ===")
print(f"Gas-generatorer: {gas_idx}")
print()

header = "Gen  |" + "".join(f" t{t:02d}" for t in range(T))

# On/off status
print("--- On/Off status (u) ---")
print(header)
for i in gas_idx:
    row = f"{i:4d} |"
    for t in range(T):
        row += f"  {int(round(u[i,t].X))}"
    print(row)

# Tænd status (v)
print("\n--- Tænd status (v) ---")
print(header)
for i in gas_idx:
    row = f"{i:4d} |"
    for t in range(T):
        val = int(round(v[i,t].X))
        row += f"  {'↑' if val==1 else '-'}"
    print(row)

# Sluk status (y)
print("\n--- Sluk status (y) ---")
print(header)
for i in gas_idx:
    row = f"{i:4d} |"
    for t in range(T):
        val = int(round(y[i,t].X))
        row += f"  {'↓' if val==1 else '-'}"
    print(row)

# Opsummering
print("\n--- Opsummering ---")
for i in gas_idx:
    timer_on   = sum(int(round(u[i,t].X)) for t in range(T))
    antal_tænd = sum(int(round(v[i,t].X)) for t in range(T))
    antal_sluk = sum(int(round(y[i,t].X)) for t in range(T))
    print(f"Gen {i}: P_bar={P_bar[i]} MW | Tændt {timer_on}/24 timer | Startet {antal_tænd}x (↑) | Slukket {antal_sluk}x (↓)")

# -------------------------
# Plot
# -------------------------

hours = list(range(T))

# Efterspørgsel (samme for alle scenarier)
demand = [sum(D[n][t] for n in range(N)) for t in range(T)]

# Gennemsnitlig vind over scenarier
wind_avg = [np.mean([w[t, xi] for xi in scenarios]) for t in range(T)]
wind_min = [max(0, np.min([w[t, xi] for xi in scenarios])) for t in range(T)]
wind_max = [np.max([w[t, xi] for xi in scenarios]) for t in range(T)]

# Gennemsnitlig kul produktion over scenarier
coal_avg = [np.mean([sum(p[i,t,xi].X for i in coal_idx) for xi in scenarios]) for t in range(T)]

# Gennemsnitlig gas produktion over scenarier
gas_avg = [np.mean([sum(p[i,t,xi].X for i in gas_idx) for xi in scenarios]) for t in range(T)]

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Usikkerhedsbånd for vind
wind_lower = [np.percentile([w[t, xi] for xi in scenarios], 5) for t in range(T)]
wind_upper = [np.percentile([w[t, xi] for xi in scenarios], 100) for t in range(T)]

ax.fill_between(hours, wind_lower, wind_upper,
                alpha=0.35,
                color='#2E7D32',
                label='Vind usikkerhedsbånd (min og max)',
                zorder=1)
# Stacked area - gennemsnit
ax.stackplot(hours,
             wind_avg,
             coal_avg,
             gas_avg,
             labels=['Vind (gns.)', 'Kul (gns.)', 'Gas (gns.)'],
             colors = ['#2E7D32', '#90A4AE', '#263238'],
             alpha=0.8)
# Demand linje
ax.plot(hours, demand,
        color='#212121', linewidth=2.5,
        linestyle='--', label='Efterspørgsel', zorder=5)

ax.set_xlabel('Time på dagen', fontsize=12)
ax.set_ylabel('MW', fontsize=12)
ax.set_title('Stokastisk produktion vs. Efterspørgsel – Gennemsnitlig vinterdag', fontsize=14)
ax.set_xticks(hours)
ax.set_xlim(0, T-1)
ax.set_ylim(0, max(demand)*1.1)
ax.set_axisbelow(True)
ax.grid(True, alpha=0.3, zorder=0)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------
# Beregn omkostninger per time (t = 1,...,T-1)
# -------------------------
hours = list(range(0, T))

# Faste omkostninger
fixed_cost = [sum(f[i] * u[i,t].X for i in range(I)) for t in range(0, T)]

# Variable omkostninger (gennemsnit over scenarier)
variable_cost = [
    np.mean([sum(c[i] * p[i,t,xi].X for i in range(I)) for xi in scenarios])
    for t in range(0, T)
]

# Opstartsomkostninger
startup_cat1 = [sum(C[i][0] * delta[i,t,0].X for i in range(I)) for t in range(0, T)]
startup_cat2 = [sum(C[i][1] * delta[i,t,1].X for i in range(I)) for t in range(0, T)]
startup_cat3 = [sum(C[i][2] * delta[i,t,2].X for i in range(I)) for t in range(0, T)]

# Penalty (gennemsnit over scenarier)
penalty_cost = [
    np.mean([
        sum(c_US * s_minus[n,t,xi].X + c_OS * s_plus[n,t,xi].X for n in range(N))
        for xi in scenarios
    ])
    for t in range(0, T)
]

# Total
total_cost = [
    fixed_cost[i] + variable_cost[i] + startup_cat1[i]
    + startup_cat2[i] + startup_cat3[i] + penalty_cost[i]
    for i in range(len(hours))
]

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.7

# Stacked bars
bars1 = ax.bar(hours, fixed_cost, bar_width,
               label='Faste omk.', color='#0D47A1', alpha=0.90)

bars2 = ax.bar(hours, variable_cost, bar_width,
               bottom=fixed_cost,
               label='Variable omk.', color='#78909C', alpha=0.90)

bars3 = ax.bar(hours, startup_cat1, bar_width,
               bottom=[fixed_cost[i] + variable_cost[i] for i in range(len(hours))],
               label='Opstart kat. 1 (varm)', color='#66BB6A', alpha=0.90)

bars4 = ax.bar(hours, startup_cat2, bar_width,
               bottom=[fixed_cost[i] + variable_cost[i] + startup_cat1[i] for i in range(len(hours))],
               label='Opstart kat. 2 (lunken)', color='#26A69A', alpha=0.90)

bars5 = ax.bar(hours, startup_cat3, bar_width,
               bottom=[fixed_cost[i] + variable_cost[i] + startup_cat1[i] + startup_cat2[i] for i in range(len(hours))],
               label='Opstart kat. 3 (kold)', color='#37474F', alpha=0.90)

bars6 = ax.bar(hours, penalty_cost, bar_width,
               bottom=[fixed_cost[i] + variable_cost[i] + startup_cat1[i] + startup_cat2[i] + startup_cat3[i]
                       for i in range(len(hours))],
               label='Penalty omk.', color='#8E24AA', alpha=0.90)

# Total som linje
ax.plot(hours, total_cost,
        color='black', linewidth=2.5,
        linestyle='--', marker='o', markersize=4,
        label='Total omk.', zorder=5)

# Layout
ax.set_xlabel('Time på dagen', fontsize=12)
ax.set_ylabel('Omkostning (kr.)', fontsize=12)
ax.set_title('Omkostningsfordeling per time – Stokastisk model', fontsize=14)
ax.set_xticks(hours)
ax.set_xlim(0.5, T - 0.5)
ax.set_ylim(0)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# -------------------------
# Print totaler
# -------------------------
print("\n--- Samlede omkostninger (t=1,...,T-1) ---")
print(f"Faste omk.:           {sum(fixed_cost):>15,.0f} kr.")
print(f"Variable omk.:        {sum(variable_cost):>15,.0f} kr.")
print(f"Opstart kat. 1:       {sum(startup_cat1):>15,.0f} kr.")
print(f"Opstart kat. 2:       {sum(startup_cat2):>15,.0f} kr.")
print(f"Opstart kat. 3:       {sum(startup_cat3):>15,.0f} kr.")
print(f"Penalty omk.:         {sum(penalty_cost):>15,.0f} kr.")
print(f"Total:                {sum(total_cost):>15,.0f} kr.")

print("\n--- Opstartskategorier ved tænding ---")
for i in range(I):
    for t in range(T):
        if v[i, t].X > 0.5:  # Generator tændes
            kat = None
            for s in range(S_i[i]):
                if delta[i, t, s].X > 0.5:
                    kat = s
            if kat == 0:
                kat_navn = "Varm"
            elif kat == 1:
                kat_navn = "Lunken"
            elif kat == 2:
                kat_navn = "Kold"
            else:
                kat_navn = "Ukendt"

            gen_type = "KUL" if i in coal_idx else "GAS"
            print(f"Gen {i:3d} [{gen_type}] | t={t:02d} | Kategori {kat}: {kat_navn} | "
                  f"Opstartsomk: {C[i][kat] if kat is not None else 0:.0f} kr.")


# -------------------------
# AVANCEREDE VISUALISERINGER
# -------------------------

# 1. Generator status heatmap (on/off, start, stop, ramping)
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Generatorernes Driftsmønstre - Vinterdag', fontsize=16, fontweight='bold')

# Heatmap: On/Off status for alle generatorer - KUN GRØN OG RØD
ax1 = axes[0, 0]
on_off_matrix = np.array([[int(round(u[i,t].X)) for t in range(T)] for i in range(I)])

# Opret custom colormap med kun to farver: rød (0) og grøn (1)
from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#8B0000', '#006400'])  # Rød og grøn

im1 = ax1.imshow(on_off_matrix, cmap=custom_cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)
ax1.set_xlabel('Time på dagen', fontsize=10)
ax1.set_ylabel('Generator ID', fontsize=10)
ax1.set_title('Tænd/Sluk Status (Grøn=Tændt, Rød=Slukket)', fontsize=11)
cbar1 = plt.colorbar(im1, ax=ax1, ticks=[0.25, 0.75])  # Placer ticks midt i hver farve
cbar1.set_label('Status', fontsize=9)
cbar1.set_ticklabels(['Slukket', 'Tændt'])
ax1.set_xticks(range(0, T, 4))
ax1.set_xticklabels([f'{h:02d}' for h in range(0, T, 4)])

# Heatmap: Opstart (v) vs Sluk (y) - GRØN for opstart, RØD for sluk
ax2 = axes[0, 1]
start_stop_matrix = np.zeros((I, T))
for i in range(I):
    for t\
            in range(T):
        if v[i, t].X > 0.5:
            start_stop_matrix[i, t] = 1  # Opstart
        elif y[i, t].X > 0.5:
            start_stop_matrix[i, t] = -1  # Sluk

# Opret custom colormap: Rød (sluk), Grå (ingen), Grøn (opstart)
from matplotlib.colors import ListedColormap
custom_cmap_start_stop = ListedColormap(['#8B0000', '#BDBDBD', '#006400'])  # Rød, Grå, Grøn

im2 = ax2.imshow(start_stop_matrix, cmap=custom_cmap_start_stop, aspect='auto',
                 interpolation='nearest', vmin=-1, vmax=1)
ax2.set_xlabel('Time på dagen', fontsize=10)
ax2.set_ylabel('Generator ID', fontsize=10)
ax2.set_title('Opstart (Grøn) vs Sluk (Rød)', fontsize=11)
cbar2 = plt.colorbar(im2, ax=ax2, ticks=[-0.66, 0, 0.66])
cbar2.set_ticklabels(['Sluk', 'Ingen ændring', 'Opstart'])
cbar2.set_label('Status', fontsize=9)
ax2.set_xticks(range(0, T, 4))
ax2.set_xticklabels([f'{h:02d}' for h in range(0, T, 4)])

# 2. Produktionsprofil for udvalgte generatorer
ax3 = axes[1, 0]
selected_gens = coal_idx[:2] + gas_idx[:2]  # Vælg 2 kul + 2 gas
all_hours = list(range(T))  # Brug alle 24 timer
for i in selected_gens:
    prod_avg = [np.mean([p[i, t, xi].X for xi in scenarios]) for t in range(T)]
    gen_name = f"Gen {i} ({'Kul' if i in coal_idx else 'Gas'})"
    ax3.plot(all_hours, prod_avg, marker='o', linewidth=2, label=gen_name, markersize=4)
ax3.set_xlabel('Time på dagen', fontsize=10)
ax3.set_ylabel('Produktion (MW)', fontsize=10)
ax3.set_title('Produktionsprofiler - Udvalgte generatorer', fontsize=11)
ax3.legend(loc='best', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, T, 4))
ax3.set_xticklabels([f'{h:02d}' for h in range(0, T, 4)])

# 3. Ramping visualisering
ax4 = axes[1, 1]
# Vælg en generator til at vise ramping
sample_gen = coal_idx[0] if coal_idx else gas_idx[0]
prod = [np.mean([p[sample_gen, t, xi].X for xi in scenarios]) for t in range(T)]
ramp_up = [max(0, prod[t] - prod[t - 1]) for t in range(1, T)]
ramp_down = [max(0, prod[t - 1] - prod[t]) for t in range(1, T)]

x_pos = np.arange(1, T)
width = 0.35
ax4.bar(x_pos - width / 2, ramp_up, width, label='Ramp Up', color='green', alpha=0.7)
ax4.bar(x_pos + width / 2, ramp_down, width, label='Ramp Down', color='red', alpha=0.7)
ax4.set_xlabel('Time (overgang)', fontsize=10)
ax4.set_ylabel('Ændring (MW/time)', fontsize=10)
ax4.set_title(f'Generator {sample_gen} - Ramping mellem timer', fontsize=11)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks(range(1, T, 2))

# 4. Opstartsomkostninger per kategori - stacked bar
ax5 = axes[2, 0]
startup_by_cat = {'Varm (Kat 1)': [], 'Lunken (Kat 2)': [], 'Kold (Kat 3)': []}
for t in range(T):
    startup_by_cat['Varm (Kat 1)'].append(sum(C[i][0] * delta[i, t, 0].X for i in range(I)))
    startup_by_cat['Lunken (Kat 2)'].append(sum(C[i][1] * delta[i, t, 1].X for i in range(I)))
    startup_by_cat['Kold (Kat 3)'].append(sum(C[i][2] * delta[i, t, 2].X for i in range(I)))

bottom = np.zeros(T)
for cat, values in startup_by_cat.items():
    ax5.bar(all_hours, values, bottom=bottom, label=cat, alpha=0.8)
    bottom += np.array(values)

ax5.set_xlabel('Time på dagen', fontsize=10)
ax5.set_ylabel('Opstartsomkostning (kr.)', fontsize=10)
ax5.set_title('Opstartsomkostninger fordelt på kategorier', fontsize=11)
ax5.legend(loc='upper right', fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')
ax5.set_xticks(range(0, T, 4))
ax5.set_xticklabels([f'{h:02d}' for h in range(0, T, 4)])

# 5. Gantt chart for generatorer (top 10 mest aktive)
ax6 = axes[2, 1]
total_active_time = [sum(int(round(u[i, t].X)) for t in range(T)) for i in range(I)]
top_gens = np.argsort(total_active_time)[-10:][::-1]

colors_gantt = ['#2E86AB' if i in coal_idx else '#A23B72' for i in top_gens]
for idx, i in enumerate(top_gens):
    for t in range(T):
        if u[i, t].X > 0.5:
            ax6.barh(idx, 1, left=t, color=colors_gantt[idx],
                     edgecolor='black', linewidth=0.5, alpha=0.8)
ax6.set_yticks(range(len(top_gens)))
ax6.set_yticklabels([f'Gen {i}' for i in top_gens])
ax6.set_xlabel('Time på dagen', fontsize=10)
ax6.set_title('Top 10 mest aktive generatorer - Driftsperioder', fontsize=11)
ax6.set_xlim(0, T)
ax6.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# -------------------------
# 6. Interaktiv visualisering (valgfrit - kræver plotly)
# -------------------------
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig_interactive = make_subplots(
        rows=3, cols=2,
        subplot_titles=('Generator Status (On/Off)', 'Produktion - Alle generatorer',
                        'Vind vs. Demand', 'Opstartsomkostninger',
                        'Ramping - Detaljeret', 'Penalty Omkostninger'),
        vertical_spacing=0.12
    )

    # Subplot 1: On/Off heatmap
    fig_interactive.add_trace(
        go.Heatmap(z=on_off_matrix, colorscale='RdYlGn',
                   x=[f't{t}' for t in range(T)], y=[f'Gen{i}' for i in range(I)],
                   name='On/Off Status'),
        row=1, col=1
    )

    # Subplot 2: Produktion over tid
    for i in range(min(5, I)):  # Vis max 5 generatorer
        prod_avg = [np.mean([p[i, t, xi].X for xi in scenarios]) for t in range(T)]
        fig_interactive.add_trace(
            go.Scatter(x=all_hours, y=prod_avg, name=f'Gen{i}', mode='lines+markers'),
            row=1, col=2
        )

    # Subplot 3: Vind vs Demand
    fig_interactive.add_trace(
        go.Scatter(x=all_hours, y=demand, name='Efterspørgsel',
                   line=dict(color='black', width=3, dash='dash')),
        row=2, col=1
    )
    fig_interactive.add_trace(
        go.Scatter(x=all_hours, y=wind_avg, name='Vind (gns.)',
                   fill='tozeroy', line=dict(color='green')),
        row=2, col=1
    )

    # Subplot 4: Opstartsomkostninger
    for cat, values in startup_by_cat.items():
        fig_interactive.add_trace(
            go.Bar(x=all_hours, y=values, name=cat),
            row=2, col=2
        )

    # Subplot 5: Ramping detaljeret
    fig_interactive.add_trace(
        go.Bar(x=x_pos, y=ramp_up, name='Ramp Up', marker_color='green'),
        row=3, col=1
    )
    fig_interactive.add_trace(
        go.Bar(x=x_pos, y=ramp_down, name='Ramp Down', marker_color='red'),
        row=3, col=1
    )

    # Subplot 6: Penalty omkostninger
    fig_interactive.add_trace(
        go.Scatter(x=all_hours[1:], y=penalty_cost, name='Penalty',
                   fill='tozeroy', line=dict(color='purple')),
        row=3, col=2
    )

    fig_interactive.update_layout(height=900, showlegend=True,
                                  title_text="Interaktiv Dashboard - Unit Commitment")
    fig_interactive.show()

    print("\n✅ Interaktiv visualisering genereret med Plotly")

except ImportError:
    print("\n⚠️ Plotly ikke installeret. Spring over interaktiv visualisering.")
    print("Installer med: pip install plotly")

# -------------------------
# 7. Udskrift af detaljeret tidslinje for hver generator
# -------------------------
print("\n" + "=" * 80)
print("DETALJERET TIDSLINJE FOR HVER GENERATOR")
print("=" * 80)

for i in range(I):
    if total_active_time[i] > 0:  # Kun generatorer der har været tændt
        gen_type = "KUL" if i in coal_idx else "GAS"
        print(f"\nGenerator {i:3d} [{gen_type}] | Max: {P_bar[i]:.0f} MW | Min: {P_underline[i]:.0f} MW")
        print("-" * 70)

        timeline = []
        current_status = None
        status_start = 0

        for t in range(T):
            status = "🟢 TÆNDT" if u[i, t].X > 0.5 else "🔴 SLUKKET"
            if status != current_status:
                if current_status is not None:
                    timeline.append((status_start, t - 1, current_status))
                current_status = status
                status_start = t

        if current_status is not None:
            timeline.append((status_start, T - 1, current_status))

        for start, end, status in timeline:
            if "TÆNDT" in status:
                prod_avg = np.mean([p[i, start, xi].X for xi in scenarios])
                print(f"  {status} | Timer {start:02d}-{end:02d} | Gns. produktion: {prod_avg:.1f} MW")
            else:
                print(f"  {status} | Timer {start:02d}-{end:02d}")

        # Vis opstarter
        startups = [t for t in range(T) if v[i, t].X > 0.5]
        if startups:
            print(f"  📈 Opstarter ved timer: {', '.join([f't{t:02d}' for t in startups])}")
            for t in startups:
                for s in range(S_i[i]):
                    if delta[i, t, s].X > 0.5:
                        kat_navn = ["VARM", "LUNKEN", "KOLD"][s]
                        print(f"     → Opstart kategori: {kat_navn} | Omkostning: {C[i][s]:.0f} kr.")

# -------------------------
# 8. Produktionsmix over tid (cirkeldiagrammer)
# -------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Produktionsmix ved udvalgte tidspunkter', fontsize=14, fontweight='bold')

selected_times = [0, 6, 12, 18, 20, 23]  # Morgen, middag, aften, nat
for idx, t in enumerate(selected_times):
    row = idx // 3
    col = idx % 3

    wind_at_t = wind_avg[t]
    coal_at_t = coal_avg[t]
    gas_at_t = gas_avg[t]

    sizes = [wind_at_t, coal_at_t, gas_at_t]
    labels = ['Vind', 'Kul', 'Gas/Damp']
    colors = ['#4CAF50', '#795548', '#FF9800']
    explode = (0.05, 0, 0)

    axes[row, col].pie(sizes, explode=explode, labels=labels, colors=colors,
                       autopct='%1.1f%%', shadow=True, startangle=90)
    axes[row, col].set_title(f'Time {t:02d}: {wind_at_t + coal_at_t + gas_at_t:.0f} MW total')

plt.tight_layout()
plt.show()

print("\n✅ Alle visualiseringer er færdige!")

# =============================================================================
# INDSÆT DETTE DIREKTE EFTER m.optimize() OG DINE EKSISTERENDE PRINTS
# Ingen funktion – bruger bare de variabler der allerede er defineret
# =============================================================================

import json, webbrowser, os

hours = list(range(T))

# ── Generatordata ─────────────────────────────────────────────────────────────
coal_gens_js, gas_gens_js = [], []
for i in range(I):
    u_vals = [int(round(u[i, t].X)) for t in range(T)]
    v_vals = [int(round(v[i, t].X)) for t in range(T)]
    y_vals = [int(round(y[i, t].X)) for t in range(T)]
    p_vals = [round(float(np.mean([p[i, t, xi].X for xi in scenarios])), 1) for t in range(T)]
    entry = {
        "id":    i,
        "pbar":  float(P_bar[i]),
        "label": f"Gen {i} ({'Kul' if i in coal_idx else 'Gas'})",
        "u": u_vals,
        "v": v_vals,
        "y": y_vals,
        "p": p_vals,
    }
    (coal_gens_js if i in coal_idx else gas_gens_js).append(entry)

# ── Efterspørgsel og vind ─────────────────────────────────────────────────────
demand_js   = [float(sum(D[n][t] for n in range(N))) for t in range(T)]
wind_avg_js = [float(np.mean([w[t, xi] for xi in scenarios])) for t in range(T)]
wind_min_js = [float(np.min( [w[t, xi] for xi in scenarios])) for t in range(T)]
wind_max_js = [float(np.max( [w[t, xi] for xi in scenarios])) for t in range(T)]

# ── Omkostninger per time ─────────────────────────────────────────────────────
fixed_cost_js   = [float(sum(f[i] * u[i, t].X for i in range(I))) for t in range(T)]
var_cost_js     = [float(np.mean([sum(c[i] * p[i, t, xi].X for i in range(I)) for xi in scenarios])) for t in range(T)]
start_warm_js   = [float(sum(C[i][0] * delta[i, t, 0].X for i in range(I) if S_i[i] > 0)) for t in range(T)]
start_luke_js   = [float(sum(C[i][1] * delta[i, t, 1].X for i in range(I) if S_i[i] > 1)) for t in range(T)]
start_cold_js   = [float(sum(C[i][2] * delta[i, t, 2].X for i in range(I) if S_i[i] > 2)) for t in range(T)]
penalty_cost_js = [
    float(np.mean([sum(c_US * s_minus[n, t, xi].X + c_OS * s_plus[n, t, xi].X
                       for n in range(N)) for xi in scenarios]))
    for t in range(T)
]
total_obj_js = float(sum(
    fixed_cost_js[t] + var_cost_js[t] + start_warm_js[t] +
    start_luke_js[t] + start_cold_js[t] + penalty_cost_js[t]
    for t in range(T)
))

data_js = {
    "T": T, "hours": hours,
    "coalGens": coal_gens_js, "gasGens": gas_gens_js,
    "demand": demand_js, "windAvg": wind_avg_js,
    "windMin": wind_min_js, "windMax": wind_max_js,
    "fixedCost": fixed_cost_js, "varCost": var_cost_js,
    "startWarm": start_warm_js, "startLuke": start_luke_js,
    "startCold": start_cold_js, "penaltyCost": penalty_cost_js,
    "totalObj": total_obj_js,
}

# ── HTML ──────────────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="da">
<head>
<meta charset="UTF-8">
<title>SUC Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#f5f5f3;color:#1a1a1a;padding:24px}}
  h1{{font-size:20px;font-weight:500;margin-bottom:20px}}
  .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
  .metric{{background:#fff;border:0.5px solid #ddd;border-radius:10px;padding:14px 18px}}
  .metric-label{{font-size:12px;color:#777;margin-bottom:4px}}
  .metric-value{{font-size:22px;font-weight:500}}
  .tabs{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}}
  .tab-btn{{background:none;border:0.5px solid #ccc;border-radius:8px;padding:6px 14px;
            font-size:13px;cursor:pointer;color:#555;transition:background .15s}}
  .tab-btn.active{{background:#e8e8e8;color:#111;border-color:#999}}
  .tab-content{{display:none}}
  .tab-content.active{{display:block}}
  .card{{background:#fff;border:0.5px solid #ddd;border-radius:12px;padding:20px}}
  .legend{{display:flex;flex-wrap:wrap;gap:14px;margin-bottom:12px;font-size:12px;color:#555}}
  .ld{{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:4px;vertical-align:middle}}
  .gen-row{{display:flex;align-items:center;gap:5px;margin-bottom:5px;font-size:12px}}
  .gen-label{{width:95px;color:#666;flex-shrink:0;font-size:11px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .cell{{width:23px;height:23px;border-radius:4px;display:inline-flex;align-items:center;
         justify-content:center;font-size:10px;font-weight:500;flex-shrink:0}}
  .on{{background:#185FA5;color:#B5D4F4}}
  .off{{background:#f0f0ee;color:#bbb;border:0.5px solid #ddd}}
  .start{{background:#1D9E75;color:#9FE1CB}}
  .stop{{background:#D85A30;color:#F5C4B3}}
  .sec{{font-size:11px;color:#999;margin:12px 0 6px}}
  .total-bar{{background:#f0f0ee;border-radius:8px;padding:10px 16px;margin-top:14px;
             font-size:13px;color:#444;display:flex;justify-content:space-between}}
</style>
</head>
<body>
<h1>SUC Optimerings-dashboard</h1>
<div class="metrics">
  <div class="metric"><div class="metric-label">Generatorer i alt</div><div class="metric-value" id="m-gen"></div></div>
  <div class="metric"><div class="metric-label">Kul-turbiner</div><div class="metric-value" id="m-coal"></div></div>
  <div class="metric"><div class="metric-label">Gas/Damp-turbiner</div><div class="metric-value" id="m-gas"></div></div>
  <div class="metric"><div class="metric-label">Total forventet omk.</div><div class="metric-value" id="m-obj"></div></div>
</div>
<div class="tabs">
  <button class="tab-btn active" onclick="showTab('s')">On/Off status</button>
  <button class="tab-btn" onclick="showTab('r')">Ramping &amp; produktion</button>
  <button class="tab-btn" onclick="showTab('c')">Omkostninger</button>
  <button class="tab-btn" onclick="showTab('b')">Energibalance</button>
</div>
<div id="s" class="tab-content active card">
  <div class="legend">
    <span><span class="ld" style="background:#185FA5"></span>Tændt</span>
    <span><span class="ld" style="background:#f0f0ee;border:0.5px solid #ddd"></span>Slukket</span>
    <span><span class="ld" style="background:#1D9E75"></span>↑ Tændes (v=1)</span>
    <span><span class="ld" style="background:#D85A30"></span>↓ Slukkes (y=1)</span>
  </div>
  <div class="sec">Kul-generatorer (P̄ ≥ 300 MW)</div>
  <div id="coal-grid"></div>
  <div class="sec" style="margin-top:16px">Gas/Damp-generatorer</div>
  <div id="gas-grid"></div>
</div>
<div id="r" class="tab-content card">
  <div class="legend" id="ramp-legend"></div>
  <div style="position:relative;height:360px"><canvas id="rampChart"></canvas></div>
</div>
<div id="c" class="tab-content card">
  <div class="legend">
    <span><span class="ld" style="background:#1565C0"></span>Faste omk.</span>
    <span><span class="ld" style="background:#FF9800"></span>Variable omk.</span>
    <span><span class="ld" style="background:#66BB6A"></span>Opstart varm</span>
    <span><span class="ld" style="background:#FFA726"></span>Opstart lunken</span>
    <span><span class="ld" style="background:#EF5350"></span>Opstart kold</span>
    <span><span class="ld" style="background:#9C27B0"></span>Penalty</span>
  </div>
  <div style="position:relative;height:320px"><canvas id="costChart"></canvas></div>
  <div class="total-bar">
    <span>Total forventet omkostning (alle timer)</span>
    <strong id="total-label"></strong>
  </div>
</div>
<div id="b" class="tab-content card">
  <div class="legend">
    <span><span class="ld" style="background:#4CAF50"></span>Vind (gns.)</span>
    <span><span class="ld" style="background:#795548"></span>Kul (gns.)</span>
    <span><span class="ld" style="background:#FF9800"></span>Gas/Damp (gns.)</span>
    <span><span class="ld" style="background:#111"></span>-- Efterspørgsel</span>
  </div>
  <div style="position:relative;height:340px"><canvas id="balChart"></canvas></div>
</div>
<script>
const D={json.dumps(data_js, ensure_ascii=False)};
const allGens=D.coalGens.concat(D.gasGens);
document.getElementById('m-gen').textContent=allGens.length;
document.getElementById('m-coal').textContent=D.coalGens.length;
document.getElementById('m-gas').textContent=D.gasGens.length;
document.getElementById('m-obj').textContent=(D.totalObj/1e6).toFixed(2)+' Mkr.';
document.getElementById('total-label').textContent=
  D.totalObj.toLocaleString('da-DK',{{maximumFractionDigits:0}})+' kr.';

function buildGrid(gens,cid){{
  const wrap=document.getElementById(cid);
  const hr=document.createElement('div');hr.className='gen-row';
  hr.innerHTML='<span class="gen-label"></span>'+
    D.hours.map(t=>`<span class="cell" style="font-size:9px;color:#aaa">${{t}}</span>`).join('');
  wrap.appendChild(hr);
  gens.forEach(g=>{{
    const row=document.createElement('div');row.className='gen-row';
    let cells=`<span class="gen-label" title="P̄=${{g.pbar}} MW">${{g.label}}</span>`;
    for(let t=0;t<D.T;t++) {{
  let cls='off', txt='-';
  if      (g.v[t]===1) {{ cls='start'; txt='↑'; }}
  else if (g.y[t]===1) {{ cls='stop';  txt='↓'; }}
  else if (g.u[t]===1) {{ cls='on';    txt='•'; }}
  cells += `<span class="cell ${{cls}}" title="t=${{t}} | ${{g.p[t]}} MW">${{txt}}</span>`;
}}
    row.innerHTML=cells;wrap.appendChild(row);
  }});
}}
buildGrid(D.coalGens,'coal-grid');
buildGrid(D.gasGens,'gas-grid');

const built={{}};
function showTab(id){{
  document.querySelectorAll('.tab-content').forEach(e=>e.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(e=>e.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.tab-btn').forEach(b=>{{
    if(b.getAttribute('onclick').includes("'"+id+"'"))b.classList.add('active');
  }});
  if(!built[id]){{buildChart(id);built[id]=true;}}
}}

const COLORS=['#185FA5','#0C447C','#1D9E75','#0F6E56','#BA7517','#854F0B','#A32D2D','#533AB7','#3B6D11','#D85A30'];

function buildChart(id){{
  if(id==='r'){{
    const leg=document.getElementById('ramp-legend');
    allGens.forEach((g,i)=>leg.innerHTML+=
      `<span><span class="ld" style="background:${{COLORS[i%COLORS.length]}}"></span>${{g.label}}</span>`);
    new Chart(document.getElementById('rampChart'),{{
      type:'line',
      data:{{labels:D.hours,datasets:allGens.map((g,i)=>{{return{{
        label:g.label,data:g.p,
        borderColor:COLORS[i%COLORS.length],backgroundColor:COLORS[i%COLORS.length]+'22',
        borderWidth:2,pointRadius:3,tension:0.3,fill:false
      }}}})}},
      options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{
          x:{{title:{{display:true,text:'Time'}},ticks:{{font:{{size:11}}}}}},
          y:{{title:{{display:true,text:'MW (gns. over scenarier)'}},ticks:{{font:{{size:11}}}}}}
        }}
      }}
    }});
  }}
  if(id==='c'){{
    new Chart(document.getElementById('costChart'),{{
      type:'bar',
      data:{{labels:D.hours,datasets:[
        {{label:'Faste',    data:D.fixedCost,  backgroundColor:'#1565C0cc'}},
        {{label:'Variable', data:D.varCost,    backgroundColor:'#FF9800cc'}},
        {{label:'Varm',     data:D.startWarm,  backgroundColor:'#66BB6Acc'}},
        {{label:'Lunken',   data:D.startLuke,  backgroundColor:'#FFA726cc'}},
        {{label:'Kold',     data:D.startCold,  backgroundColor:'#EF5350cc'}},
        {{label:'Penalty',  data:D.penaltyCost,backgroundColor:'#9C27B0cc'}},
      ]}},
      options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{
          x:{{stacked:true,ticks:{{font:{{size:11}}}}}},
          y:{{stacked:true,title:{{display:true,text:'kr.'}},
             ticks:{{font:{{size:11}},callback:v=>v>=1000?Math.round(v/1000)+'k':v}}}}
        }}
      }}
    }});
  }}
  if(id==='b'){{
    const coalP=D.hours.map(t=>D.coalGens.reduce((s,g)=>s+g.p[t],0));
    const gasP =D.hours.map(t=>D.gasGens.reduce((s,g)=>s+g.p[t],0));
    new Chart(document.getElementById('balChart'),{{
      type:'line',
      data:{{labels:D.hours,datasets:[
        {{label:'Vind',data:D.windAvg,backgroundColor:'#4CAF5055',borderColor:'#4CAF50',fill:'origin',tension:0.3,borderWidth:1.5}},
        {{label:'Kul', data:coalP,    backgroundColor:'#79554855',borderColor:'#795548',fill:'-1',    tension:0.3,borderWidth:1.5}},
        {{label:'Gas', data:gasP,     backgroundColor:'#FF980055',borderColor:'#FF9800',fill:'-1',    tension:0.3,borderWidth:1.5}},
        {{label:'Efterspørgsel',data:D.demand,borderColor:'#111',borderWidth:2.5,
          borderDash:[6,3],fill:false,tension:0.3,pointRadius:0}}
      ]}},
      options:{{responsive:true,maintainAspectRatio:false,
        plugins:{{legend:{{display:false}}}},
        scales:{{
          x:{{ticks:{{font:{{size:11}}}}}},
          y:{{title:{{display:true,text:'MW'}},ticks:{{font:{{size:11}}}}}}
        }}
      }}
    }});
  }}
}}
</script>
</body></html>"""

output_path = "suc_dashboard.html"
with open(output_path, "w", encoding="utf-8") as fh:
    fh.write(html)
abs_path = os.path.abspath(output_path)
print(f"\nDashboard gemt → {abs_path}")
webbrowser.open(f"file://{abs_path}")



import matplotlib.pyplot as plt
import numpy as np

hours = np.arange(24)

dayahead_vals = wind['ForecastDayAhead'].head(24).values
current_vals  = wind['ForecastCurrent'].head(24).values
noise_vals    = (wind['ForecastCurrent'] - wind['ForecastDayAhead']).head(24).values

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(hours, dayahead_vals, marker='o', linewidth=2,
        color='#1565C0', label='ForecastDayAhead (MW)')
ax.plot(hours, current_vals, marker='s', linewidth=2,
        color='#2E7D32', label='ForecastCurrent (MW)')
ax.bar(hours, noise_vals, alpha=0.35, color='#D84315',
       label=r'Støj $\varepsilon_t$ = Current − DayAhead (MW)')
ax.axhline(0, color='#888', linewidth=0.8, linestyle='--')

ax.set_xlabel('Time på dagen')
ax.set_ylabel('MW')
ax.set_xticks(hours)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')

plt.title(r'ForecastDayAhead, ForecastCurrent og støj $\varepsilon_t$')
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

hours = np.arange(24)

dayahead_vals = wind['ForecastDayAhead'].head(24).values
current_vals  = wind['ForecastCurrent'].head(24).values
noise_vals    = current_vals - dayahead_vals   # ε_t = Current - DayAhead

fig, ax = plt.subplots(figsize=(13, 6))

# Day-ahead og Current
ax.plot(hours, dayahead_vals, marker='o', linewidth=2,
        color='#1565C0', label=r'ForecastDayAhead$_t$ (deterministisk prognose)')
ax.plot(hours, current_vals, marker='s', linewidth=2,
        color='#2E7D32', label=r'ForecastCurrent$_t$ (realiseret vindproduktion)')

# Støj som søjler
ax.bar(hours, noise_vals, bottom=dayahead_vals, alpha=0.35, color='#D84315', width=0.6,
       label=r'$\varepsilon_t$ = Current$_t$ $-$ DayAhead$_t$ (prognosefejl)')

# Simulerede scenarier
for xi in scenarios:
    scenario_wind = [w[t, xi] for t in range(24)]
    ax.plot(hours, scenario_wind, linewidth=0.7, alpha=0.4,
            color='#7B1FA2', linestyle='--')

# Dummy til legend for scenarier
ax.plot([], [], linewidth=0.7, color='#7B1FA2', linestyle='--',
        alpha=0.8, label=r'Simulerede scenarier $\boldsymbol{\xi}(\omega_i)$ (LHS + Weibull)')

ax.set_xlabel('Time på dagen', fontsize=12)
ax.set_ylabel('MW', fontsize=12)
ax.set_xticks(hours)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=10)
ax.set_title(r'Prognosefejl $\hat{\varepsilon_t}$ og simulerede scenarier – DK2 vinter',
             fontsize=13)

plt.tight_layout()
plt.show()


print(wind_min)
print(wind_max)

print(f"DayAhead t=3: {dayahead_day[3][0]:.2f}")
print(f"wind_min t=3: {wind_min[3]:.2f}")
print(f"wind_max t=3: {wind_max[3]:.2f}")