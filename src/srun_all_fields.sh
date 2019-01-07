#!/bin/bash
#
#SBATCH -J lc_builder
#SBATCH --output=/home/jmartinez/HiTS/logs/diff_lc.o%A
#SBATCH --error=/home/jmartinez/HiTS/errors/diff_lc.%j.err
#SBATCH --ntasks 1
#SBATCH --mem-per-cpu=6000
#

srun --exclusive -n 1 -c 1 python features_calc_leftraru.py -F $1 -C all -b g

echo "Finished Test Job"
exit 0
