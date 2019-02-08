#!/bin/bash

## Job Name
#SBATCH -J std_lc

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH --output=/home/jmartinez/HiTS/logs/std_lc.%A.%a.log
#SBATCH --error=/home/jmartinez/HiTS/errors/std_lc.%A.%a.err

# Queue name (partition name)
#SBATCH --partition=slims
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1

## Job Array tasks (from 1 to 60)
#SBATCH --array=1-50
#SBATCH --mem=2000

## Email notification (all events)
##SBATCH -M jorgemarpa@ug.uchile.cl
##SBATCH --mail-type=NONE

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

echo "$SLURM_ARRAY_TASK_ID"

FILELIST="/home/jmartinez/HiTS/info/fields_Blind15A.txt"

echo "$FILELIST"

field=`head -n $SLURM_ARRAY_TASK_ID $FILELIST | tail -n 1 | cut -f 1 -d" "`

echo $field

## read epoch.txt file with epochs
${PYTHON} get_lim_var_lc.py -F $field
EXIT_CODE=$?

## finishing tasks

echo "Exit code: $EXIT_CODE"

##  - remove temporary files
##  - handle errors
