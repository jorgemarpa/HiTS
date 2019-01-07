#!/bin/bash

## Job Name
#SBATCH -J background_leftraru.slurm

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH --output=/home/jmartinez/HiTS/logs/back.%A.%a.log
#SBATCH --error=/home/jmartinez/HiTS/errors/back.%A.%a.err

# Processors:
# Number of MPI tasks
#SBATCH -n 1
##SBATCH --ntasks-per-node=1
#
# Number of cores per task
#SBATCH -c 1

# Queue name (partition name)
#SBATCH --partition=slims

## Job Array tasks (from 1 to 5)
#SBATCH --array=1-60

## Email notification (all events)
##SBATCH -M jorgemarpa@ug.uchile.cl
##SBATCH --mail-type=all

# parametrize your variables
PYTHON=python

# get access to the SLURM task id variable
export SLURM_JOBID               ; # real job id
export SLURM_ARRAY_JOB_ID        ; # parent job id (originator job)
export SLURM_ARRAY_TASK_ID       ; # task id

if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
        # not under slurm environment.
        # assuming only one input file
        SLURM_ARRAY_TASK_ID=1
fi

# make the toolchain available
module load astro

# list the modules loaded
module list

# include your home bin directory into the PATH
export PATH=$HOME/bin:$PATH

# export the OpenMP number of thread requested
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [[ -z "$OMP_NUM_THREADS" ]]; then
     # no openmp requested to slurm. setting to default
     OMP_NUM_THREADS=1
fi

echo "Number of thread per task : $OMP_NUM_THREADS"

# write the hostname where we are running
echo "Hostname: `hostname`"

## My Script ##

export PYTHONPATH=`pwd`/test_omp:$PYTHONPATH

## read region.txt file with name of regions

echo "$SLURM_ARRAY_TASK_ID"

FILELIST="/home/jmartinez/HiTS/info/ccds.txt"

echo "$FILELIST"

FILENAME=`head -n $SLURM_ARRAY_TASK_ID $FILELIST | tail -n 1 | cut -f 1 -d" "`

echo "$FILENAME"

${PYTHON} background_leftraru.py $1 $FILENAME
EXIT_CODE=$?

## finishing tasks

echo "Exit code: $EXIT_CODE"

##  - remove temporary files
##  - handle errors
