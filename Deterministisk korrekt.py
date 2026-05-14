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
m.setParam('Threads', 1)        # Reproducerbarhed
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

print("\n--- s_minus og s_plus per time ---")
for t in range(T):
    s_minus_t = sum(s_minus[n, t].X for n in range(N))
    s_plus_t = sum(s_plus[n, t].X for n in range(N))

    print(f"t={t:02d}: s_minus={s_minus_t:.2f} | s_plus={s_plus_t:.2f}")
print("\n--- Opstarter per generator og kategori ---")


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
        row += f" {p[i,t].X:5.0f}"
    print(row)

# Opsummering
print("\n--- Opsummering ---")
for i in coal_idx:
    timer_on = sum(int(round(u[i,t].X)) for t in range(T))
    total_p  = sum(p[i,t].X for t in range(T))
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
# Data til plot af demand
# -------------------------
hours = list(range(T))

# Samlet efterspørgsel per time
demand = [sum(D[n][t] for n in range(N)) for t in range(T)]

# Vind per time
wind = [w[t] for t in range(T)]

# Kul produktion per time
coal_prod = [sum(p[i,t].X for i in coal_idx) for t in range(T)]

# Gas/damp produktion per time
gas_prod = [sum(p[i,t].X for i in gas_idx) for t in range(T)]

# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# Stacked area plot
ax.stackplot(hours,
             wind,
             coal_prod,
             gas_prod,
             labels=['Vind', 'Kul', 'Gas'],
             colors=['#2E7D32', '#90A4AE', '#263238'],
             alpha=0.8)

# Demand linje
ax.plot(hours, demand,
        color='#212121',
        linewidth=2.5,
        linestyle='--',
        label='Efterspørgsel',
        zorder=5)

# Formatering
ax.set_xlabel('Time på dagen', fontsize=12)
ax.set_ylabel('MW', fontsize=12)
ax.set_title('Produktion og Efterspørgsel – Gennemsnitlig vinterdag', fontsize=14)
ax.set_xticks(hours)
ax.set_xlim(0, T-1)
ax.set_ylim(0)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# -------------------------
# Data til plot af pris - deterministisk
# -------------------------
hours = list(range(T))

# Faste omkostninger per time
fixed_cost_det = [sum(f[i] * u[i,t].X for i in range(I)) for t in range(T)]

# Variable omkostninger per time
variable_cost_det = [sum(c[i] * p[i,t].X for i in range(I)) for t in range(T)]

# Opstartsomkostninger per kategori per time
startup_cat1_det = [sum(C[i][0] * delta[i,t,0].X for i in range(I) if S_i[i] > 0) for t in range(T)]
startup_cat2_det = [sum(C[i][1] * delta[i,t,1].X for i in range(I) if S_i[i] > 1) for t in range(T)]
startup_cat3_det = [sum(C[i][2] * delta[i,t,2].X for i in range(I) if S_i[i] > 2) for t in range(T)]

# Penalty per time
penalty_cost_det = [
    sum(c_US * s_minus[n,t].X + c_OS * s_plus[n,t].X for n in range(N))
    for t in range(T)
]

# Total per time
total_cost_det = [
    fixed_cost_det[t] + variable_cost_det[t] + startup_cat1_det[t] +
    startup_cat2_det[t] + startup_cat3_det[t] + penalty_cost_det[t]
    for t in range(T)
]

# -------------------------
# Plot pris - deterministisk
# -------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bar_width = 0.7
hours = list(range(1, T))  # 1,...,23
bottom = np.zeros(len(hours))

for values, label, color in [
    (fixed_cost_det,    'Faste omk.',             '#0D47A1'),
    (variable_cost_det, 'Variable omk.',           '#78909C'),
    (startup_cat1_det,  'Opstart kat. 1 (varm)',   '#66BB6A'),
    (startup_cat2_det,  'Opstart kat. 2 (lunken)', '#26A69A'),
    (startup_cat3_det,  'Opstart kat. 3 (kold)',   '#37474F'),
    (penalty_cost_det,  'Penalty omk.',            '#8E24AA'),
]:
    ax.bar(hours, values[1:], bar_width, bottom=bottom, label=label, color=color, alpha=0.85)
    bottom += np.array(values[1:])

ax.plot(hours, total_cost_det[1:],
        color='black', linewidth=2.5,
        linestyle='--', marker='o', markersize=4,
        label='Total omk.', zorder=5)

ax.set_xlabel('Time på dagen', fontsize=12)
ax.set_ylabel('Omkostning (kr.)', fontsize=12)
ax.set_title('Omkostningsfordeling per time – Deterministisk model', fontsize=14)
ax.set_xticks(hours)
ax.set_xlim(-0.5, T - 0.5)
ax.set_ylim(0)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# -------------------------
# Print totaler
# -------------------------
print("\n--- Samlede omkostninger (deterministisk) ---")
print(f"Faste omk.:           {sum(fixed_cost_det):>15,.0f} kr.")
print(f"Variable omk.:        {sum(variable_cost_det):>15,.0f} kr.")
print(f"Opstart kat. 1:       {sum(startup_cat1_det):>15,.0f} kr.")
print(f"Opstart kat. 2:       {sum(startup_cat2_det):>15,.0f} kr.")
print(f"Opstart kat. 3:       {sum(startup_cat3_det):>15,.0f} kr.")
print(f"Penalty omk.:         {sum(penalty_cost_det):>15,.0f} kr.")
print(f"Total:                {sum(total_cost_det):>15,.0f} kr.")

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


# Kør KUN én gang, ikke inde i et loop
t = 0
tændt_kapacitet = sum(P_bar[i] for i in range(I) if DT0[i] == 0)
netto_demand    = sum(D[n][0] for n in range(N)) - w[0]
print(f"Kapacitet ved t=0: {tændt_kapacitet:.0f} MW")
print(f"Netto efterspørgsel: {netto_demand:.0f} MW")
print(f"Dækker: {'JA' if tændt_kapacitet >= netto_demand else 'NEJ – mangler ' + str(round(netto_demand - tændt_kapacitet)) + ' MW'}")

# =============================================================================
# DETERMINISTISK DASHBOARD - Indsæt efter m.optimize() og eksisterende prints
# =============================================================================

import json, webbrowser, os

hours = list(range(T))

# ── Generatordata ─────────────────────────────────────────────────────────────
coal_gens_js, gas_gens_js = [], []
for i in range(I):
    u_vals = [int(round(u[i, t].X)) for t in range(T)]
    v_vals = [int(round(v[i, t].X)) for t in range(T)]
    y_vals = [int(round(y[i, t].X)) for t in range(T)]
    p_vals = [round(float(p[i, t].X), 1) for t in range(T)]  # Ingen scenarier i deterministisk
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
wind_js     = [float(w[t]) for t in range(T)]  # Deterministisk vind (gennemsnit)

# ── Omkostninger per time ─────────────────────────────────────────────────────
fixed_cost_js   = [float(sum(f[i] * u[i, t].X for i in range(I))) for t in range(T)]
var_cost_js     = [float(sum(c[i] * p[i, t].X for i in range(I))) for t in range(T)]
start_warm_js   = [float(sum(C[i][0] * delta[i, t, 0].X for i in range(I) if S_i[i] > 0)) for t in range(T)]
start_luke_js   = [float(sum(C[i][1] * delta[i, t, 1].X for i in range(I) if S_i[i] > 1)) for t in range(T)]
start_cold_js   = [float(sum(C[i][2] * delta[i, t, 2].X for i in range(I) if S_i[i] > 2)) for t in range(T)]
penalty_cost_js = [
    float(sum(c_US * s_minus[n, t].X + c_OS * s_plus[n, t].X for n in range(N)))
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
    "demand": demand_js, "wind": wind_js,  # Kun én vindværdi (ingen min/max i deterministisk)
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
<title>DUC Deterministisk Dashboard</title>
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
<h1>DUC Deterministisk Dashboard - Gennemsnitlig vinterdag</h1>
<div class="metrics">
  <div class="metric"><div class="metric-label">Generatorer i alt</div><div class="metric-value" id="m-gen"></div></div>
  <div class="metric"><div class="metric-label">Kul-turbiner</div><div class="metric-value" id="m-coal"></div></div>
  <div class="metric"><div class="metric-label">Gas/Damp-turbiner</div><div class="metric-value" id="m-gas"></div></div>
  <div class="metric"><div class="metric-label">Total omkostning</div><div class="metric-value" id="m-obj"></div></div>
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
    <span>Total omkostning (alle timer)</span>
    <strong id="total-label"></strong>
  </div>
</div>
<div id="b" class="tab-content card">
  <div class="legend">
    <span><span class="ld" style="background:#4CAF50"></span>Vind</span>
    <span><span class="ld" style="background:#795548"></span>Kul</span>
    <span><span class="ld" style="background:#FF9800"></span>Gas/Damp</span>
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
          y:{{title:{{display:true,text:'MW'}},ticks:{{font:{{size:11}}}}}}
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
        {{label:'Vind',data:D.wind,backgroundColor:'#4CAF5055',borderColor:'#4CAF50',fill:'origin',tension:0.3,borderWidth:1.5}},
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

output_path = "duc_deterministic_dashboard.html"
with open(output_path, "w", encoding="utf-8") as fh:
    fh.write(html)
abs_path = os.path.abspath(output_path)
print(f"\nDashboard gemt → {abs_path}")
webbrowser.open(f"file://{abs_path}")