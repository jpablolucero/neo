#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=2
#SBATCH -p devel
#SBATCH -t 2:0:0
#SBATCH -A bw16K021
#
# overview of SBATCH parameters:
# -N 2/--nodes=2 number of nodes
# -n 8/--ntasks=8 total number of mpi processes (2x4)
# --ntasks-per-nodes=4 ...
# -t 1-0/--time=1-0 max. runtime/walltime (time formats see 'man sbatch')
# -p devel/--partition=devel run in partition (=queue) devel
# -A account/--account=account charge resources to this account/RV
# more options:
# --begin=<timespec>  start the job at the time specified f.e.
# --begin=now+1hour  start in 1 hour from now (see man sbatch)
# -d afterok:<jobid> start the job after another job with jobid <jobid> has
# successfully finished (see man sbatch)
# --reservation=<reservationname> (from sinfo -T): run in reservation
# -w|--nodelist=<nodes> run on named nodes
# -s|--share share the node with other jobs (hits queue)
# -i|--input connect standard input and
# -e|--error connect standard error and
# -o|--output standard output, see man sbatch

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Current working directory is `pwd`"
echo "LD_LIBRARY_PATH is $LD_LIBRARY_PATH"
echo "current mpicc is $(which mpicc)"
echo "tmpdir is ${TMPDIR}"
set|grep I_MPI
set|grep OMPI

undate=$(date +%Y-%m-%d_%R)

srun --mpi=pmi2 ./neo 3  0 0 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -1 0 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -2 0 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -3 0 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -4 0 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -5 0 3000 100 500 0

srun --mpi=pmi2 ./neo 3  0 1.E-5 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -1 1.E-5 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -2 1.E-5 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -3 1.E-5 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -4 1.E-5 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -5 1.E-5 3000 100 500 0

srun --mpi=pmi2 ./neo 3  0 100 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -1 100 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -2 100 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -3 100 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -4 100 3000 100 500 0
srun --mpi=pmi2 ./neo 3 -5 100 3000 100 500 0

srun --mpi=pmi2 ./neo 3  0 1.E-5 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -1 1.E-5 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -2 1.E-5 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -3 1.E-5 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -4 1.E-5 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -5 1.E-5 3000 100 500 1

srun --mpi=pmi2 ./neo 3  0 100 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -1 100 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -2 100 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -3 100 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -4 100 3000 100 500 1
srun --mpi=pmi2 ./neo 3 -5 100 3000 100 500 1


echo "Program finished with exit code $? at: `date`"
