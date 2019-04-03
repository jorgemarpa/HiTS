#!/bin/bash

## Job Name
#SBATCH -J do_cat_leftraru.slurm

## Job Output (stdout + stderr) file
## %A = parent job id
## %a = task id
## %j = real job id
#SBATCH --output=/home/jmartinez/HiTS/logs/do_cat.%A.%a.log
#SBATCH --error=/home/jmartinez/HiTS/errors/do_cat.%A.%a.err

# Processors:
# Number of MPI tasks
#SBATCH -n 1
# Number of cores per task
#SBATCH -c 1
#SBATCH --mem-per-cpu=6000

# Queue name (partition name)
#SBATCH --partition=slims

## Job Array tasks (from 1 to 60)
#SBATCH --array=1-60

## Email notification (all events)
##SBATCH -M jorgemarpa@ug.uchile.cl
##SBATCH --mail-type=NONE

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

CHIPS="/home/jmartinez/HiTS/info/ccds.txt"

echo "$CHIPS"

CCD=`head -n $SLURM_ARRAY_TASK_ID $CHIPS | tail -n 1 | cut -f 1 -d" "`

echo "$1" "$CCD"

echo 'Getting epochs for filter u...'
${PYTHON} get_epochs_leftraru.py -F $1 -B u
echo 'Getting epochs for filter g...'
${PYTHON} get_epochs_leftraru.py -F $1 -B g
echo 'Getting epochs for filter r...'
${PYTHON} get_epochs_leftraru.py -F $1 -B r
echo 'Getting epochs for filter i...'
${PYTHON} get_epochs_leftraru.py -F $1 -B i

##############################################################
## Here start the pipeline

# if [[ "$1" == "Blind13A"* ]]; then
#       echo 'Checking background images for epoch 02...'
#       echo '   Filter u'
#       ${PYTHON} background_leftraru.py $1 $CCD u
#       EXIT_CODE=$?
#       echo "Exit code: $EXIT_CODE"
#
#       echo "###########################################################"
#       echo "###########################################################"
#
#       echo 'Stacking CCDs images...'
#       ${PYTHON} stack_leftraru.py $1 $CCD SUM u
#       EXIT_CODE=$?
#       echo "Exit code: $EXIT_CODE"
#
#       echo "###########################################################"
#       echo "###########################################################"
#
#       echo 'Running SEx in stacked images...'
#       ${PYTHON} sextractor_stack_leftraru.py $1 $CCD
#       EXIT_CODE=$?
#       echo "Exit code: $EXIT_CODE"
#       echo "###########################################################"
#       echo "###########################################################"
# fi


echo 'Running SEx in single epochs images...'
${PYTHON} sextractor_epoch_leftraru.py $1 $CCD
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "###########################################################"


echo 'Calculating astrometric solution...'
if [[ "$1" == "Blind13A"* ]]; then
      echo 'Filter u'
      ${PYTHON} matchstars_leftraru.py -F $1 -C $CCD -b u -a 02 -p 02
      EXIT_CODE=$?
      echo "Exit code: $EXIT_CODE"
      echo "###########################################################"

elif [[ "$1" == "Blind14A"* ]]; then
      echo 'Filter g'
      ${PYTHON} matchstars_leftraru.py -F $1 -C $CCD -b g -a 02 -p 02
      EXIT_CODE=$?
      echo "Exit code: $EXIT_CODE"
      echo "###########################################################"

elif [[ "$1" == "Blind15A"* ]]; then
      echo 'Filter g'
      ${PYTHON} matchstars_leftraru.py -F $1 -C $CCD -b g -a 02 -p 02
      EXIT_CODE=$?
      echo "Exit code: $EXIT_CODE"
      echo "###########################################################"
      echo 'Filter r'
      ${PYTHON} matchstars_leftraru.py -F $1 -C $CCD -b r -a 02 -p 01
      EXIT_CODE=$?
      echo "Exit code: $EXIT_CODE"
      echo "###########################################################"
      echo 'Filter i'
      ${PYTHON} matchstars_leftraru.py -F $1 -C $CCD -b i -a 02 -p 02
      EXIT_CODE=$?
      echo "Exit code: $EXIT_CODE"
      echo "###########################################################"
      echo 'Filter u'
      ${PYTHON} matchstars_leftraru.py -F $1 -C $CCD -b u -a 02 -p 02
      EXIT_CODE=$?
      echo "Exit code: $EXIT_CODE"
      echo "###########################################################"
fi
echo "###########################################################"
echo "###########################################################"
echo "###########################################################"

##############################################################

echo 'SCAMP for astrometry in reference epoch...'
${PYTHON} scamp_for_HiTS_leftraru.py -F $1 -C $CCD -e 02
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "###########################################################"
echo "###########################################################"
echo "###########################################################"


##############################################################

echo 'Transforming quantities...'
${PYTHON} trans_catal_leftraru.py -F $1 -C $CCD
EXIT_CODE=$?
echo "Exit code: $EXIT_CODE"
echo "###########################################################"
echo "###########################################################"
echo "###########################################################"

##############################################################

# echo 'Creating index file to build LC...'
# echo '   Filter u'
# ${PYTHON} create_lightcurves_leftraru.py -F $1 -C $CCD -b u
# EXIT_CODE=$?
# echo "Exit code: $EXIT_CODE"
# echo "###########################################################"
# echo '   Filter g'
# ${PYTHON} create_lightcurves_leftraru.py -F $1 -C $CCD -b g
# EXIT_CODE=$?
# echo "Exit code: $EXIT_CODE"
# echo "###########################################################"
# echo '   Filter r'
# ${PYTHON} create_lightcurves_leftraru.py -F $1 -C $CCD -b r
# EXIT_CODE=$?
# echo "Exit code: $EXIT_CODE"
# echo "###########################################################"
# echo '   Filter i'
# ${PYTHON} create_lightcurves_leftraru.py -F $1 -C $CCD -b i
# EXIT_CODE=$?
# echo "Exit code: $EXIT_CODE"
# echo "###########################################################"
# echo "###########################################################"
# echo "###########################################################"
#
# ##############################################################
#
# echo 'Creating photometric table...'
# if [[ "$1" == "Blind15A"* ]]; then
#       echo '   Filters ugri'
#       ${PYTHON} make_table_leftraru.py -F $1 -C $CCD -o 3 -b ugri
#       EXIT_CODE=$?
#       echo "Exit code: $EXIT_CODE"
#       echo "###########################################################"
# elif [[ "$1" == "Blind14A"* ]]; then
#       echo '   Filter g'
#       ${PYTHON} make_table_leftraru.py -F $1 -C $CCD -o 3 -b g
#       EXIT_CODE=$?
#       echo "Exit code: $EXIT_CODE"
#       echo "###########################################################"
# elif [[ "$1" == "Blind13A"* ]]; then
#       echo '   Filter u'
#       ${PYTHON} make_table_leftraru.py -F $1 -C $CCD -o 3 -b u
#       EXIT_CODE=$?
#       echo "Exit code: $EXIT_CODE"
#       echo "###########################################################"
# fi
#
#
#
# echo 'Saving LC to txt files and compress'
# if [[ "$1" == "Blind15A"* ]]; then
#       echo '   Filter g'
#       ${PYTHON} read_lightcurves_leftraru.py -F $1 -C $CCD -o 15 -b g -c True
#       EXIT_CODE=$?
# elif [[ "$1" == "Blind14A"* ]]; then
#       echo '   Filter g'
#       ${PYTHON} read_lightcurves_leftraru.py -F $1 -C $CCD -o 15 -b g -c True
#       EXIT_CODE=$?
# elif [[ "$1" == "Blind13A"* ]]; then
#       echo '   Filter u'
#       ${PYTHON} read_lightcurves_leftraru.py -F $1 -C $CCD -o 15 -b u -c True
#       EXIT_CODE=$?
# fi
# echo "Exit code: $EXIT_CODE"
# echo "###########################################################"
# echo "###########################################################"
# echo "###########################################################"

# echo 'Calculating light-curves features'
# if [[ "$1" == "Blind15A"* ]]; then
#       echo '   Filter g'
#       ${PYTHON} features_calc_leftraru.py -F $1 -C $CCD -b g -o 15
#       EXIT_CODE=$?
# elif [[ "$1" == "Blind14A"* ]]; then
#       echo '   Filter g'
#       ${PYTHON} features_calc_leftraru.py -F $1 -C $CCD -b g -o 15
#       EXIT_CODE=$?
# elif [[ "$1" == "Blind13A"* ]]; then
#       echo '   Filter u'
#       ${PYTHON} features_calc_leftraru.py -F $1 -C $CCD -b u -o 15
#       EXIT_CODE=$?
# fi
# echo "Exit code: $EXIT_CODE"

##############################################################
