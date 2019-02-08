#!/bin/bash

## Job Name
#SBATCH -J lightcurve_rrly_leftraru.slurm

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH --output=/home/jmartinez/HiTS/logs/RRLy.%A.%a.log
#SBATCH --error=/home/jmartinez/HiTS/errors/RRLy.%A.%a.err

## Job Array, 20 parallels jobs
#SBATCH --partition=slims
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
##SBATCH --mail-user=user@gmail.com
##SBATCH --mail-type=ALL

# parametrize your variables
PYTHON=python

# get access to the SLURM task id variable
export SLURM_JOBID               ; # real job id
export SLURM_ARRAY_JOB_ID        ; # parent job id (originator job)
export SLURM_ARRAY_TASK_ID       ; # task id

# make the toolchain available
module load astro

# list the modules loaded
module list

# include your home bin directory into the PATH
export PATH=$HOME/bin:$PATH

# write the hostname where we are running
echo "Hostname: `hostname`"

## My Script ##

export PYTHONPATH=`pwd`/test_omp:$PYTHONPATH

## read epoch.txt file with epochs

${PYTHON} lightcurve_rrly_leftraru.py
EXIT_CODE=$?

## finishing tasks

echo "Exit code: $EXIT_CODE"

##  - remove temporary files
##  - handle errors
