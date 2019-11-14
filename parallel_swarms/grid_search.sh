#!/bin/bash

#PBS -A CMDA_Cap_18
#PBS -W group_list=newriver
#PBS -q normal_q
#PBS -l nodes=1
#PBS -l proc=20
#PBS -l walltime=04:00:00

source ./loadmods.sh
module load cmake
cp CMakeLists.txt CMakeListsOLD.txt
cp CMakeListsARC.txt CMakeLists.txt
make clean
cmake .

mpirun -np $PBS_NP ./single_optimal_parallel

exit;
