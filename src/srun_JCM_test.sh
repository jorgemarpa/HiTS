#!/bin/bash
#
#SBATCH -J test_job
#SBATCH --output=/home/jmartinez/HiTS/logs/test_job.o%A
#SBATCH --ntasks 5
#SBATCH --mem-per-cpu=15000
#

echo "MultiStep Test Job"

NUM_STEPS=31
MAX_CONCURRENT_STEPS=$SLURM_NTASKS
ip=$(ip -o -4 addr list eno1 | awk '{print $4}' | cut -d/ -f1)

seconds=0

influxd &

seq -f "N%g" 1 $NUM_STEPS | xargs -n 1 -P $MAX_CONCURRENT_STEPS  srun --exclusive -n 1 -c 1 python influxdb_HiTS.py -t True -i $ip -F $1 -c

#seq -f "S%g" 1 $NUM_STEPS | xargs -n 1 -P $MAX_CONCURRENT_STEPS  srun --exclusive -n 1 -c 1 python influxdb_HiTS.py -t True -i $ip -F $1 -c

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "Finished Test Job"
exit 0
