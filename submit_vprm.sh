#!/bin/bash -l
#
#SBATCH --job-name=vprm
#SBATCH --output=slurmjob.%j.o
#SBATCH --error=slurmjob.%j.e
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=03:00:00
#SBATCH --constraint=gpu
#SBATCH --account=s862

#=======START===========

#	make sure the desired programming environment is available
module load daint-mc
#module switch PrgEnv-cray PrgEnv-gnu
module load analytics

#   just to make sure: set unlimited stack size, should be the default on daint
ulimit -s unlimited

/scratch/snx3000/ochsnerd/conda-env/bin/python3 make_vprm_emi.py
