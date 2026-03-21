#!/bin/sh
module load namd3/2025-12-04/multicore-CUDA
namd3 +p2  +devices $2 $1 +stdout log.$1
