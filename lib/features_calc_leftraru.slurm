#!/bin/bash
## Job Name
#SBATCH -J Features

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH --output=/home/jmartinez/HiTS/logs/feat.%A.%a.log
#SBATCH --error=/home/jmartinez/HiTS/errors/feat.%A.%a.err

## Job Array, 20 parallels jobs
#SBATCH --partition=slims
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 1
#SBATCH --mem-per-cpu=7000
#SBATCH --array=1-58
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

FILELIST="/home/jmartinez/HiTS/info/galaxy_field_14A_15A.txt"
echo "$FILELIST"
field=`head -n $SLURM_ARRAY_TASK_ID $FILELIST | tail -n 1`
echo "$field"

# CHIPS="/home/jmartinez/HiTS/info/ccds.txt"
# echo "$CHIPS"
# CCD=`head -n $SLURM_ARRAY_TASK_ID $CHIPS | tail -n 1 | cut -f 1 -d" "`
# echo "$CCD"

echo ${PYTHON} features_calc_leftraru.py -F $field -C all -b g -o 10 -a $1
${PYTHON} features_calc_leftraru.py -F $field -C all -b g -o 10 -a $1
EXIT_CODE=$?

## finishing tasks

echo "Exit code: $EXIT_CODE"

##  - remove temporary files
##  - handle errors
