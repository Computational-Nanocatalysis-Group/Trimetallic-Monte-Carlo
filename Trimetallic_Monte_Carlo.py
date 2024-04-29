# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 19:59:23 2024

@author: Arravind Subramanian, Mikhail V. Polyinski, Mathan K. Eswaran, Sergey M. Kozlov
"""

from ase.visualize import view
from ase.io import write, read
import random
import numpy as np
from ase.data import covalent_radii as CR
import pandas as pd
import time 
from ase.cluster import Octahedron
import matplotlib.pyplot as plt
import pickle
import os
import csv
from ase.units import kB

def get_coordination_numbers(atoms, covalent_percent=1.25):
    """cn returns the coordination numbers of each atom, bonded returns the 
        atoms which atom of index [i] is coordinated with. You can aslo tune
        the covalent percent to obtain correctly the bonded information of the 
        nanoparticle. Check bonded information for a nanoparticle before 
        proceeding to MC simulations.
        
        It can be done using the following commands
            from collections import Counter
            cn = get_coordination_numbers(ASE_generated_nanoparticle)
            print(Counter(cn[0]))
        Typical output for 79-atom Octahedron is "Counter({6: 24, 9: 24, 7: 12, 12: 19})
    """

    available_distances = np.divide(atoms.get_all_distances(mic=True), covalent_percent)
    numbers = atoms.numbers
    cn = []
    C_radii = np.take(CR, numbers)
    bonded_atoms = []
    indices = list(range(len(atoms)))
    for a in indices:
        sub_bonded = []
        for b in indices:
            if a == b:
                continue
            if (C_radii[a] + C_radii[b]) >= available_distances[a,b]:
                sub_bonded.append(b)
        bonded_atoms.append(sub_bonded)
    for bonded_lists in bonded_atoms:
        cn.append(len(bonded_lists))
    return cn, bonded_atoms

# seed values can be initialized depending on your simulation 
# random.seed(SET_VALUE)

def shuffle_nano(symbols,length,metal_A,metal_B,metal_C):
    """Can be used to shuffle one randomly chosen metal atom A with another
        randomly chosen metal atom B or C only one time in a single move
    """
    Met_positions = symbols.copy()
    Metal_1_s,Metal_2_s = random.sample(set([metal_A,metal_B,metal_C]), 2) 
    pos_A = [atom for atom in range(length) if Met_positions[atom] == Metal_1_s]
    pos_B = [atom for atom in range(length) if Met_positions[atom] == Metal_2_s]       
    i = random.choice(pos_A)
    j = random.choice(pos_B)
    Met_positions[i],Met_positions[j] = Met_positions[j],Met_positions[i]
    return Met_positions

def dynamic_shuffle_nano(symbols,length,metal_A,metal_B,metal_C,loop):
    """Can be used to shuffle one randomly chosen metal atom A with another
        randomly chosen metal atom B or C multiple time in a single move
    """
    Met_positions = symbols.copy()
    for i in range(loop):
        Metal_1_s,Metal_2_s = random.sample(set([metal_A,metal_B,metal_C]), 2) 
        pos_A = [atom for atom in range(length) if Met_positions[atom] == Metal_1_s]
        pos_B = [atom for atom in range(length) if Met_positions[atom] == Metal_2_s]       
        i = random.choice(pos_A)
        j = random.choice(pos_B)
        Met_positions[i],Met_positions[j] = Met_positions[j],Met_positions[i]
    return Met_positions

def get_descriptors_nano(symbols,cn,metal_1,metal_2,metal_3):
    """This function extracts the structural information of the trimetallic 
        nanoparticle during each step. This particular function works well
        for truncated Octahedral nanoaprticles, but depending on the type of 
        structure used one can modify this fucntion by changing the coordination
        number in the "cn[0][i]" sections in the first for loop.
        
        CAUTION : Before executing this loop, please make sure your nanoparticle
        is crystalline and not deformed. This can be checked by using the following 
        codes.
            from collections import Counter
            cn = get_coordination_numbers(ASE_generated_nanoparticle)
            print(Counter(cn[0]))
        Typical output for 79-atom Octahedron is "Counter({6: 24, 9: 24, 7: 12, 12: 19})"    
    """
    
    A_B_bonds_f = 0
    B_C_bonds_f = 0
    A_C_bonds_f = 0
    A_corners_f = 0
    A_edges_f   = 0
    A_terrace_f = 0
    B_corners_f = 0
    B_edges_f   = 0
    B_terrace_f = 0
    C_corners_f = 0
    C_edges_f   = 0
    C_terrace_f = 0    
    
    for i in range(len(cn[0])):
        ref_atom = symbols[i]
        if ref_atom == metal_1 and cn[0][i] == 6:
            A_corners_f += 1
        elif ref_atom == metal_1 and cn[0][i] == 7:  
            A_edges_f += 1
        elif ref_atom == metal_1 and cn[0][i] == 9:  
            A_terrace_f += 1    
        if ref_atom == metal_2 and cn[0][i] == 6:
            B_corners_f += 1
        elif ref_atom == metal_2 and cn[0][i] == 7:  
            B_edges_f += 1
        elif ref_atom == metal_2 and cn[0][i] == 9:  
            B_terrace_f += 1     
        if ref_atom == metal_3 and cn[0][i] == 6:
            C_corners_f += 1
        elif ref_atom == metal_3 and cn[0][i] == 7:  
            C_edges_f += 1
        elif ref_atom == metal_3 and cn[0][i] == 9:  
            C_terrace_f += 1             

    for atom in range(len(cn[0])):
        for pair in range(len(cn[1][atom])):
            if (symbols[atom],symbols[cn[1][atom][pair]]) == (metal_1, metal_2) or (symbols[atom],symbols[cn[1][atom][pair]]) == (metal_2,metal_1):
                A_B_bonds_f += 1
            if (symbols[atom],symbols[cn[1][atom][pair]]) == (metal_2, metal_3) or (symbols[atom],symbols[cn[1][atom][pair]]) == (metal_3,metal_2):
                B_C_bonds_f += 1                
            if (symbols[atom],symbols[cn[1][atom][pair]]) == (metal_1, metal_3) or (symbols[atom],symbols[cn[1][atom][pair]]) ==  (metal_3,metal_1):
                A_C_bonds_f += 1
                
    return A_B_bonds_f/2, B_C_bonds_f/2, A_C_bonds_f/2, A_corners_f, A_edges_f, A_terrace_f, B_corners_f, B_edges_f, B_terrace_f, C_corners_f, C_edges_f, C_terrace_f

def Monte_Carlo(MC_inputs,NP_structure,composition_metal_1,composition_metal_2,des_metal_1,des_metal_2,other_metal_3):
    """For the provided input this function acts the Monte-Carlo engine"""
    Inputs = MC_inputs.copy()

    Energies =  []
    step =      []
    Req_energies = []
    structures = []

    N_steps       = Inputs[0]      
    Temperature   = Inputs[1]    
    intercept     = Inputs[2] 
    Des_A_B_bonds = Inputs[3] 
    Des_B_C_bonds = Inputs[4] 
    Des_A_C_bonds = Inputs[5] 
    Des_A_corners = Inputs[6] 
    Des_A_edges   = Inputs[7] 
    Des_A_terrace = Inputs[8] 
    Des_B_corners = Inputs[9] 
    Des_B_edges   = Inputs[10] 
    Des_B_terrace = Inputs[11] 
    
    nano = NP_structure.copy()
    symbols = nano.get_chemical_symbols()
    Metal_1 = des_metal_1
    Metal_2 = des_metal_2
    Metal_3 = other_metal_3
    Metal_1_conc = composition_metal_1
    Metal_2_conc = composition_metal_2
    Metal_3_conc = len(symbols) - Metal_1_conc - Metal_2_conc
    for i in range(Metal_1_conc):
        symbols[i] = Metal_1
    for i in range(Metal_2_conc):
        symbols[Metal_1_conc + i] = Metal_2
    for i in range(Metal_3_conc):
        symbols[Metal_1_conc + Metal_2_conc + i] = Metal_3  
        
    length = len(nano)    
        
    seeded_symbols = dynamic_shuffle_nano(symbols,length,Metal_1,Metal_2,Metal_3,1000)
    
    nano.set_chemical_symbols(seeded_symbols)

    symbols = seeded_symbols.copy()

    # view(nano)
    
    cn = get_coordination_numbers(nano)
        
    
    struc_des = get_descriptors_nano(symbols,cn,metal_1=Metal_1,metal_2=Metal_2,metal_3=Metal_3)
    Initial_Energy = intercept + (Des_A_B_bonds*struc_des[0]) + (Des_B_C_bonds*struc_des[1]) + (Des_A_C_bonds*struc_des[2]) + (Des_A_corners*struc_des[3]) +\
        (Des_A_edges*struc_des[4]) + (Des_A_terrace*struc_des[5]) + (Des_B_corners*struc_des[6]) + (Des_B_edges*struc_des[7]) + (Des_B_terrace*struc_des[8])
    Energies.append(Initial_Energy)
    Req_energies.append(Initial_Energy)
    step.append(1)
    i = 2
    struct_count = 1
    name = 'nano'+'_'+str(0)+'.xyz'
    write(name,nano,format='xyz')
    structures.append(symbols)
     
    ener_path = 'nano_ener.csv'
    results_path = 'ML_nano.csv'    

    with open(results_path,'w') as file_1, \
            open(ener_path,'w') as file_2:
                    
        struct_data = csv.writer(file_1, delimiter=',',lineterminator='\n',)  
        ener_data = csv.writer(file_2, delimiter=',',lineterminator='\n',)
        
        struct_data.writerow(['Structure','Energy','A-B_bonds','B-C_bonds','A-C_bonds','A_corners','A_edges','A_terrace','B_corners','B_edges','B_terrace','C_corners','C_edges','C_terrace'])
        ener_data.writerow(['step','Energies'])
        
        struct_data.writerow([0,Initial_Energy,struc_des[0],struc_des[1],struc_des[2],struc_des[3],struc_des[4],struc_des[5],struc_des[6],struc_des[7],struc_des[8],struc_des[9],struc_des[10],struc_des[11]])
        ener_data.writerow([1,Initial_Energy])
        t0= time.perf_counter()
        while i < N_steps:
            symbols = symbols
            new_symbols = shuffle_nano(symbols,length,Metal_1,Metal_2,Metal_3)
            struc_des = get_descriptors_nano(new_symbols,cn,metal_1=Metal_1,metal_2=Metal_2,metal_3=Metal_3)
            Actual_Energy = intercept + (Des_A_B_bonds*struc_des[0]) + (Des_B_C_bonds*struc_des[1]) + (Des_A_C_bonds*struc_des[2]) + (Des_A_corners*struc_des[3]) +\
                (Des_A_edges*struc_des[4]) + (Des_A_terrace*struc_des[5]) + (Des_B_corners*struc_des[6]) + (Des_B_edges*struc_des[7]) + (Des_B_terrace*struc_des[8])
            if np.exp(-(Actual_Energy - Energies[-1])/(8.617330337217213e-05*Temperature)) > random.random():
                symbols = new_symbols.copy()
                if Actual_Energy != Energies[-1]:
                    ener_data.writerow([i,Actual_Energy])
                    Energies.append(Actual_Energy)
                    if Actual_Energy < Req_energies[-1]:
                        struct_data.writerow([struct_count,Actual_Energy,struc_des[0],struc_des[1],struc_des[2],struc_des[3],struc_des[4],struc_des[5],struc_des[6],struc_des[7],struc_des[8],struc_des[9],struc_des[10],struc_des[11]])
                        Req_energies.append(Actual_Energy)
                        structures.append(symbols.copy())
                        nano.set_chemical_symbols(symbols.copy())
                        name = 'nano'+'_'+str(struct_count)+'.xyz'
                        write(name,nano,format='xyz')
                        struct_count += 1
            else:
                symbols = symbols
                
            i += 1
    
        ener_data.writerow([i,Energies[-1]])  
        t1 = time.perf_counter()
        print("Time elapsed: ", t1 - t0)
    
    ener = pd.read_csv(ener_path)
    step_data = ener['step'].to_list()
    Energy_data = ener['Energies'].to_list()
    fig = plt.figure(figsize=(8,6), dpi = 80)
    plt.step(step_data, Energy_data)
    plt.xlabel("steps")
    plt.ylabel("Energy (eV)")
    figure_path = 'fig.jpg'
    fig.savefig(figure_path, dpi = fig.dpi)
    
    results = pd.read_csv(results_path)

    return results, ener, structures, fig

def Monte_Carlo_ini(MC_inputs,NP_structure,composition_metal_1,composition_metal_2,des_metal_1,des_metal_2,other_metal_3):
    """This fucntion will be used in the generation of initial set of 
        132 structures. Since the descriptors obtained are scalable to 
        bigger nanoaprticle we recommend to use nanoparticles which has
        less than 150 atoms and 1 million MC steps for less cpu time
    """
    Inputs = MC_inputs.copy()

    Energies =  []
    A_B_bonds = []
    B_C_bonds = []
    A_C_bonds = []
    A_corners = []
    A_edges =   []
    A_terrace = []
    B_corners = []
    B_edges =   []
    B_terrace = []
    C_corners = []
    C_edges =   []
    C_terrace = []
    step =      []
    Req_energies = []
    structures = []

    N_steps       = Inputs[0]      
    Temperature   = Inputs[1]    
    intercept     = Inputs[2] 
    Des_A_B_bonds = Inputs[3] 
    Des_B_C_bonds = Inputs[4] 
    Des_A_C_bonds = Inputs[5] 
    Des_A_corners = Inputs[6] 
    Des_A_edges   = Inputs[7] 
    Des_A_terrace = Inputs[8] 
    Des_B_corners = Inputs[9] 
    Des_B_edges   = Inputs[10] 
    Des_B_terrace = Inputs[11] 
    
    nano = NP_structure.copy()
    symbols = nano.get_chemical_symbols()
    Metal_1 = des_metal_1
    Metal_2 = des_metal_2
    Metal_3 = other_metal_3
    Metal_1_conc = composition_metal_1
    Metal_2_conc = composition_metal_2
    Metal_3_conc = len(symbols) - Metal_1_conc - Metal_2_conc
    for i in range(Metal_1_conc):
        symbols[i] = Metal_1
    for i in range(Metal_2_conc):
        symbols[Metal_1_conc + i] = Metal_2
    for i in range(Metal_3_conc):
        symbols[Metal_1_conc + Metal_2_conc + i] = Metal_3  
        
    length = len(nano)    
        
    seeded_symbols = dynamic_shuffle_nano(symbols,length,Metal_1,Metal_2,Metal_3,1000)
    
    nano.set_chemical_symbols(seeded_symbols)

    # view(nano)

    symbols = seeded_symbols.copy()
    
    cn = get_coordination_numbers(nano)
        
    
    struc_des = get_descriptors_nano(symbols,cn,metal_1=Metal_1,metal_2=Metal_2,metal_3=Metal_3)
    Initial_Energy = intercept + (Des_A_B_bonds*struc_des[0]) + (Des_B_C_bonds*struc_des[1]) + (Des_A_C_bonds*struc_des[2]) + (Des_A_corners*struc_des[3]) +\
        (Des_A_edges*struc_des[4]) + (Des_A_terrace*struc_des[5]) + (Des_B_corners*struc_des[6]) + (Des_B_edges*struc_des[7]) + (Des_B_terrace*struc_des[8])
    Energies.append(Initial_Energy)
    Req_energies.append(Initial_Energy)
    A_B_bonds.append(struc_des[0])
    B_C_bonds.append(struc_des[1])
    A_C_bonds.append(struc_des[2])
    A_corners.append(struc_des[3])
    A_edges.append(struc_des[4])
    A_terrace.append(struc_des[5])
    B_corners.append(struc_des[6])
    B_edges.append(struc_des[7])
    B_terrace.append(struc_des[8])
    C_corners.append(struc_des[9])
    C_edges.append(struc_des[10])
    C_terrace.append(struc_des[11])
    step.append(1)
    i = 2
    structures.append(symbols)
    t0= time.perf_counter()
    while i < N_steps:
        symbols = symbols
        new_symbols = shuffle_nano(symbols,length,Metal_1,Metal_2,Metal_3)
        struc_des = get_descriptors_nano(new_symbols,cn,metal_1=Metal_1,metal_2=Metal_2,metal_3=Metal_3)
        Actual_Energy = intercept + (Des_A_B_bonds*struc_des[0]) + (Des_B_C_bonds*struc_des[1]) + (Des_A_C_bonds*struc_des[2]) + (Des_A_corners*struc_des[3]) +\
            (Des_A_edges*struc_des[4]) + (Des_A_terrace*struc_des[5]) + (Des_B_corners*struc_des[6]) + (Des_B_edges*struc_des[7]) + (Des_B_terrace*struc_des[8])
        if np.exp(-(Actual_Energy - Energies[-1])/(kB*Temperature)) > random.random():
            symbols = new_symbols.copy()
            if Actual_Energy != Energies[-1]:
                Energies.append(Actual_Energy)
                step.append(i)
                if Actual_Energy < Req_energies[-1]:
                    Req_energies.append(Actual_Energy)
                    A_B_bonds.append(struc_des[0])
                    B_C_bonds.append(struc_des[1])
                    A_C_bonds.append(struc_des[2])
                    A_corners.append(struc_des[3])
                    A_edges.append(struc_des[4])
                    A_terrace.append(struc_des[5])
                    B_corners.append(struc_des[6])
                    B_edges.append(struc_des[7])
                    B_terrace.append(struc_des[8])
                    C_corners.append(struc_des[9])
                    C_edges.append(struc_des[10])
                    C_terrace.append(struc_des[11])
                    structures.append(symbols.copy())
        else:
            symbols = symbols
            
        i += 1

    step.append(i)
    Energies.append(Energies[-1])    
    t1 = time.perf_counter()
    print("Time elapsed: ", t1 - t0)
    fig = plt.figure(figsize=(8,6), dpi = 80)
    plt.step(step, Energies)
    plt.xlabel("steps")
    plt.ylabel("Energy (eV)")

    data = {'Energy'    : Req_energies,
            'A_B_bonds' : A_B_bonds,
            'B_C_bonds' : B_C_bonds,
            'A_C_bonds' : A_C_bonds,
            'A_corner'  : A_corners,
            'A_edge'    : A_edges,
            'A_terrace' : A_terrace,
            'B_corner'  : B_corners,
            'B_edge'    : B_edges,
            'B_terrace' : B_terrace,   
            'C_corner'  : C_corners,
            'C_edge'    : C_edges,
            'C_terrace' : C_terrace}
    
    data_ener = {'step' : step,
                  'Energies' : Energies}
        
    results = pd.DataFrame(data=data) 
    
    ener = pd.DataFrame(data=data_ener)

    return results, ener, structures, fig

def get_MC_inputs(string,MC_steps,Temp,intercept):
    """Depending on the string provided this helper function generates
        inputs for performing Monte-Carlo simulation. This function will
        be only for the case of 132 initial structures
    """
    parameters = string.split()

    MC_inputs = [0]*12
    # the above is the energetic descriptors as
    # [MC_steps,Temperature,Intercept,AB_bonds,BC_bonds,AC_bonds,A_Atom_6,A_Atom_7,A_Atom_9,B_Atom_6,B_Atom_7,B_Atom_9]
    
    MC_inputs[0] = MC_steps
    
    MC_inputs[1] = Temp
    
    MC_inputs[2] = intercept
    
    if 'bonds' in parameters:
        if parameters[0] == "high" and parameters[1] == "AB":
            MC_inputs[3] = -100
        elif parameters[0] == "low" and parameters[1] == "AB":
            MC_inputs[3] = 100
        elif parameters[0] == "high" and parameters[1] == "BC":
            MC_inputs[4] = -100
        elif parameters[0] == "low" and parameters[1] == "BC":
            MC_inputs[4] = 100
        elif parameters[0] == "high" and parameters[1] == "AC":
            MC_inputs[5] = -100
        elif parameters[0] == "low" and parameters[1] == "AC":
            MC_inputs[5] = 100            
        return MC_inputs
    else:
        if parameters[-2] == "high" and parameters[-1] == "AB":
            MC_inputs[3] = -10
        elif parameters[-2] == "low" and parameters[-1] == "AB":
            MC_inputs[3] = 10
        elif parameters[-2] == "high" and parameters[-1] == "BC":
            MC_inputs[4] = -10
        elif parameters[-2] == "low" and parameters[-1] == "BC":
            MC_inputs[4] = 10
        elif parameters[-2] == "high" and parameters[-1] == "AC":
            MC_inputs[5] = -10
        elif parameters[-2] == "low" and parameters[-1] == "AC":
            MC_inputs[5] = 10  
       
        if parameters[0] == "max":
            if parameters[2] == "6": 
                if parameters[3] == "met_A": 
                    MC_inputs[6] = -100
                elif parameters[3] == "met_B":  
                    MC_inputs[9] = -100
                elif parameters[3] == "met_C":  
                    MC_inputs[6] = 100
                    MC_inputs[9] = 100                    
            if parameters[2] == "7": 
                if parameters[3] == "met_A": 
                    MC_inputs[7] = -100
                elif parameters[3] == "met_B":  
                    MC_inputs[10] = -100
                elif parameters[3] == "met_C":  
                    MC_inputs[7] = 100
                    MC_inputs[10] = 100 
            if parameters[2] == "9": 
                if parameters[3] == "met_A": 
                    MC_inputs[8] = -100
                elif parameters[3] == "met_B":  
                    MC_inputs[11] = -100
                elif parameters[3] == "met_C":  
                    MC_inputs[8] = 100
                    MC_inputs[11] = 100                     
                    
        elif parameters[0] == "min":
            if parameters[2] == "6": 
                if parameters[3] == "met_A": 
                    MC_inputs[6] = 100
                elif parameters[3] == "met_B":  
                    MC_inputs[9] = 100
                elif parameters[3] == "met_C":  
                    MC_inputs[6] = -100
                    MC_inputs[9] = -100                    
            if parameters[2] == "7": 
                if parameters[3] == "met_A": 
                    MC_inputs[7] = 100
                elif parameters[3] == "met_B":  
                    MC_inputs[10] = 100
                elif parameters[3] == "met_C":  
                    MC_inputs[7] = -100
                    MC_inputs[10] = -100 
            if parameters[2] == "9": 
                if parameters[3] == "met_A": 
                    MC_inputs[8] = 100
                elif parameters[3] == "met_B":  
                    MC_inputs[11] = 100
                elif parameters[3] == "met_C":  
                    MC_inputs[8] = -100
                    MC_inputs[11] = -100 
            
        return MC_inputs

def run_Monte_Carlo(string,MC_steps,Temp,intercept,input_nanoparticle,comp_metal_1,comp_metal_2,Metal_1,Metal_2,Metal_3,folder_path):
    """This is another helper function which will be used to create the inital
        set of 132 structures
    """
    composition_metal_1 = comp_metal_1
    composition_metal_2 = comp_metal_2
    metal_1 = Metal_1
    metal_2 = Metal_2
    metal_3 = Metal_3
    NP_structure = input_nanoparticle
    if string == 'all combinations':
        Final_structures = []
        all_struct_des = []
        all_MC_inputs = []
        all_possibilities = get_all_possible_systems()
        for combination in all_possibilities:
            MC_ini_inputs = get_MC_inputs(combination,MC_steps,Temp,intercept)
            MC_ini_inputs_append = MC_ini_inputs.copy()
            MC_ini_inputs_append.insert(0,combination)
            all_MC_inputs.append(MC_ini_inputs_append)
            path_dir = folder_path + '/' + combination
            os.mkdir(path_dir)
            print(MC_ini_inputs_append)
            run_engine = Monte_Carlo_ini(MC_ini_inputs,NP_structure,composition_metal_1,composition_metal_2,des_metal_1=metal_1,des_metal_2=metal_2,other_metal_3=metal_3)
            results_from_MC_engine = run_engine[0]
            path_results = path_dir + '/' + 'descriptors.xlsx'
            results_from_MC_engine.to_excel(path_results)
            results_ener_from_MC_engine = run_engine[1] 
            path_ener = path_dir + '/' + 'ener.xlsx'
            results_ener_from_MC_engine.to_excel(path_ener)
            for outputs in range(len(run_engine[2])):
                save_NP = NP_structure.copy()
                save_NP.set_chemical_symbols(run_engine[2][outputs]) 
                save_name = 'nano'+'_'+str(outputs)+'.xyz'
                save_nano_path = path_dir + '/' + save_name
                write(save_nano_path,save_NP,format='xyz')
            path_fig = path_dir + '/' + 'ener_fig.jpg'    
            run_engine[3].savefig(path_fig, dpi = run_engine[3].dpi)    
            Final_structures.append(run_engine[2][-1])
            # view_NP = NP_structure.copy()
            # view_NP.set_chemical_symbols(run_engine[2][-1])    
            # view(view_NP)
            struct_des_list = run_engine[0].iloc[-1].tolist()
            struct_des_list.insert(0,combination)
            print(struct_des_list)
            print("############################################################################################")
            all_struct_des.append(struct_des_list)
            column_names = run_engine[0].columns.tolist()
        MC_input_column_names = ['Name','MC_steps','Temperature','Intercept','AB_bonds','BC_bonds','AC_bonds','A_Atom_6','A_Atom_7',
                                 'A_Atom_9','B_Atom_6','B_Atom_7','B_Atom_9']
        Descriptors_provided = pd.DataFrame(all_MC_inputs,columns=MC_input_column_names)
        column_names.insert(0,'Names')
        Obtained_data = pd.DataFrame(all_struct_des,columns=column_names)
        return Obtained_data, Descriptors_provided, Final_structures   
    else:        
        MC_ini_inputs = get_MC_inputs(string,MC_steps,Temp,intercept)
        print(MC_ini_inputs)
        run_engine = Monte_Carlo_ini(MC_ini_inputs,NP_structure,composition_metal_1,composition_metal_2,des_metal_1=metal_1,des_metal_2=metal_2,other_metal_3=metal_3)
        return run_engine

def get_all_possible_systems():    
    """ This is a string editing function which generates the names of the 
        initial set of 132 structures
    """
    limits = ['max','min']
    coordination_sites = ['6','7','9']
    metals = ['met_A','met_B','met_C']
    bonds = ['random','low AB','low BC','low AC','high AB','high BC','high AC']
    structure_types = []
    
    for i in limits:
        for j in coordination_sites:
            for k in metals:
                for l in bonds:
                    structure_types.append(i + ' ' + 'coord' + ' ' + j + ' ' + k+ ' ' + l)
    
    structure_types.append('low' + ' ' +'AB'+ ' ' + 'bonds')
    structure_types.append('high' + ' ' +'AB'+ ' ' + 'bonds')
    structure_types.append('low' + ' ' +'BC'+ ' ' + 'bonds')
    structure_types.append('high' + ' ' +'BC'+ ' ' + 'bonds')
    structure_types.append('low' + ' ' +'AC'+ ' ' + 'bonds')
    structure_types.append('high' + ' ' +'AC'+ ' ' + 'bonds')
    return structure_types

t2 = time.perf_counter()

nanoparticle = Octahedron('Pt',5,1)
nanoparticle.center(vacuum = 5)

##########  For running simulations of all possible combinations ##############

main_path = "PROVIDE PATH OF FOLDER"
# string = "all combinations" for generation all the 132 structures 
simulation = run_Monte_Carlo(string,MC_steps,Temp,intercept,input_nanoparticle,comp_metal_1,comp_metal_2,Metal_1,Metal_2,Metal_3,folder_path)
simulation[0].to_excel("PROVIDE PATH OF FOLDER/obtained_data.xlsx")
simulation[1].to_excel("PROVIDE PATH OF FOLDER/input_data.xlsx")
with open("struc_pickle_path", "wb") as fp:   #Pickling
    pickle.dump(simulation[2], fp)
with open("struc_pickle_path", "rb") as fp:   # Unpickling
    NP_structures = pickle.load(fp)  
struct_names = simulation[0]['Names'].to_list()
for i in range(len(struct_names)):
    save_NP = nanoparticle.copy()
    nanoparticle.set_chemical_symbols(NP_structures[i]) 
    save_name = struct_names[i]+'.xyz'
    save_nano_path = main_path + '/' + save_name
    write(save_nano_path,save_NP,format='xyz') 

######### End of above independent code ########## 


##########  For running imdividual simulation ##############
#########  Please comment previous lines of code from (604-619) before executing individual jobs #########

# manual_input = [MC_steps,Temperature,Intercept,AB_bonds,BC_bonds,AC_bonds,A_Atom_6,A_Atom_7,A_Atom_9,B_Atom_6,B_Atom_7,B_Atom_9]  
# manual_simulation = Monte_Carlo(manual_input,nanoparticle,26,26,'A','B','C')

######### End of above independent code ##########

t3 = time.perf_counter()
print("Time elapsed: ", t3 - t2)