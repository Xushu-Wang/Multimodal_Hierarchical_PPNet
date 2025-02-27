#!/bin/bash 

./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 0.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 1.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 10.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 100.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 1000.0 & 
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 0.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 1.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 10.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 100.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 1000.0 & 
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 0.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 1.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 10.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 100.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 1000.0 & 
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 0.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 1.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 10.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 100.0 &
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 1000.0 & 
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 0.0 &
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 1.0 &
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 10.0 &
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 100.0 &
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 1000.0 & 

