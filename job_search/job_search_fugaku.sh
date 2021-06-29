#!/bin/sh
#PJM --rsc-list "node=4000"
#PJM --rsc-list "rscunit=rscunit_ft01"
#PJM --rsc-list "rscgrp=large"
#PJM --rsc-list "elapse=08:00:00"
#PJM --mpi "max-proc-per-node=12"
#PJM -S

export OMP_NUM_THREADS=4

mpiexec -stdout-proc ./%/1000R/stdout -stderr-proc ./%/1000R/stderr ../search.out 9600 25200 5 1234
