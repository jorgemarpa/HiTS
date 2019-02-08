#!/bin/bash
#
#SBATCH -J test_job
#SBATCH --output=/home/jmartinez/HiTS/logs/drop.o%A
#SBATCH --ntasks 1
#SBATCH --mem-per-cpu=10000
#

echo "MultiStep Test Job"
ip=$(ip -o -4 addr list eno1 | awk '{print $4}' | cut -d/ -f1)

seconds=0

influxd &

srun python influxdb_HiTS.py -d True -i 192.168.50.141

duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "Finished Test Job"
exit 0
