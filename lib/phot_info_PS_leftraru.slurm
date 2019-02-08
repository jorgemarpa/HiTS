#!/bin/bash

## Job Name
#SBATCH -J phot_epoch_leftraru.slurm

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH --output=/home/jmartinez/HiTS/logs/info.%A.%a.log
#SBATCH --error=/home/jmartinez/HiTS/errors/info.%A.%a.err

## Job Array, 20 parallels jobs
#SBATCH --partition=slims
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem-per-cpu=5000
#SBATCH --array=1-20
##SBATCH --mail-user=user@gmail.com
##SBATCH --mail-type=ALL

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

# write the hostname where we are running
echo "Hostname: `hostname`"

## My Script ##

export PYTHONPATH=`pwd`/test_omp:$PYTHONPATH

## read epoch.txt file with epochs

echo "$SLURM_ARRAY_TASK_ID"

FILELIST="/home/jmartinez/HiTS/info/$1/$1_epochs_$2.txt"

echo "$FILELIST"

FILENAME=`head -n $SLURM_ARRAY_TASK_ID $FILELIST | tail -n 1 | cut -f 1 -d" "`

echo "$FILENAME"

${PYTHON} phot_info_PS_leftraru.py -F $1 -b $2 -e $FILENAME
EXIT_CODE=$?

## finishing tasks

echo "Exit code: $EXIT_CODE"

##  - remove temporary files
##  - handle errors
