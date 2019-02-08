#!/bin/bash

## Job Name
#SBATCH -J stamps

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH --output=/home/jmartinez/HiTS/logs/stamps.%A.%a.log
#SBATCH --error=/home/jmartinez/HiTS/errors/stamps.%A.%a.err

# Queue name (partition name)
#SBATCH --partition=slims
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1

## Job Array tasks (from 1 to 60)
#SBATCH --array=1-500
#SBATCH --mem=1000

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

FILELIST="/home/jmartinez/HiTS/info/NV_15A_list.txt"

echo "$FILELIST"

id=`head -n $SLURM_ARRAY_TASK_ID $FILELIST | tail -n 1 | cut -d ',' -f 1`

echo $id

## read epoch.txt file with epochs
${PYTHON} lightcurve_pix_leftraru_ff.py -b g -m id -p stamp -i $id
EXIT_CODE=$?

## finishing tasks

echo "Exit code: $EXIT_CODE"

##  - remove temporary files
##  - handle errors
