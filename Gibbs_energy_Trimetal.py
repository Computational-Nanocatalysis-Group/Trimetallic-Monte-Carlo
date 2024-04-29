# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:07:44 2024

@author: Arravind Subramanian, Mikhail V. Polyinski, Mathan K. Eswaran, Sergey M. Kozlov
"""

import numpy as np
import pandas as pd
import math
from ase.units import kB

def fact(a):
    out = 0.5*np.log(2*math.pi*a) + a*(np.log(a) - np.log(math.e))
    return out

N_a = 'TOTAL NO OF A ATOMS'
N_b = 'TOTAL NO OF B ATOMS'
N_c = 'TOTAL NO OF C ATOMS'
N = N_a + N_b + N_c
T = 'TEMPERATURE OF SIMULATION'

main_path = "PATH TO ENER FILE"

E_data = pd.read_csv(main_path)
len_value = int(round(len(E_data)/2,0))
E = np.array(E_data['Energies'][len_value:], dtype=np.float64)
steps = np.array(E_data['step'][len_value:])
high_E = min(list(np.array(E_data['Energies'], dtype=np.float64)))
ln_N_states = fact(N) - fact(N_a) - fact(N_b) - fact(N_c)
gaussian = []
for i in range(len(E)):
    gaussian.append(np.exp(-(E[i] - high_E)/(kB*T)))
step_diff = []
for j in range(len(steps)):
    if j == 0:
        step_diff.append(1)  
    else:
        step_diff.append(steps[j]-steps[j-1])
total_gaussian = []
for i in range(len(step_diff)):
    total_gaussian.append(step_diff[i]*gaussian[i])
N_steps = np.sum(step_diff)    
q = np.sum(total_gaussian)/N_steps   
ln_Q = np.log(q) + ln_N_states 
G = -kB*T*(np.log(q) + ln_N_states) + high_E
print(G)
    














