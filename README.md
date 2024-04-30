# Trimetallic-Monte-Carlo
This repository contains a simple python code for performing Monte-Carlo simulation of Trimetallic nanoparticles

This software is capable of performing three major functions:

- Generate initial set of 132 structures.
- Perform Monte-Carlo simulations of nanoparticles of various size, shape and composition.
- Determine Helmholtz free energy of the nanoparticle.

##  Prerequisites

This package requires:

- [NumPy](https://numpy.org/)
- [ase](https://wiki.fysik.dtu.dk/ase/)
- [pandas](https://pandas.pydata.org/)

# Usage

## To generate the initial set of 132 structures

- Provide the nanoparticle shape, composition and the elements in the run_Monte_Carlo() function.

#### Structure string usage

- The function get_all_possible_systems() generates the name strings of the 132 initial strctures.
- Depending on the purpose, one can choose to generate certain structures alone by calling the string name in the run_Monte_Carlo() function.
- Calling the string "all combinations" in the run_Monte_Carlo() function will generate all the 132 structures.

#### CAUTION 

- While generating the 132 structure, one can use less than 150 atoms to reduce computational cost. The energetic descriptors obtained from smaller nanoparticles can also be used for bigger nanoparticles.

#### Output

The run_Monte_Carlo() function generates three main ouputs.
- An excel file containing all inputs and another excel file containing structural informaltion of lowest energy structures in each case.
- The archtypal structures will be saved in pickle file and also be saved in .xyz format
- An example usage of the code is provided in the lines 604-619 of the Trimetallic_Monte_Carlo.py file.

## To run Monte_Carlo simulations

- Provide the energetic descriptors, MC steps and Temperature in a seperate list in the format given below
- manual_input = [MC_steps,Temperature,Intercept,AB_bonds,BC_bonds,AC_bonds,A_Atom_6,A_Atom_7,A_Atom_9,B_Atom_6,B_Atom_7,B_Atom_9]
- Call the Monte_Carlo() function by providing the above input, nanoparticle, composition and the elements to run the simulation.

#### Output

- Ouputs including the energy parameters at each accepted step and the lowest energy structure will be generated during the execution as "nano_ener.csv" and "ML_nano.csv" respectively
- The low energy structures will also be saved in .xyz format

## To find free energy of Trimetal system.

- Perform Monte-Carlo siulation at high temperatures for a minimum of 10 million steps depending on the size of the nanoparticle
- Use the obtained "nano_ener.csv" file as input to the Gibbs_energy_Trimetal.py and provide the composition.
- The script will generate the free energy of the nanoparticle.

## License

This code is released under the NUS License.
  
