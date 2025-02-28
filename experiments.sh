#!/bin/bash 

sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 10.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 100.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.0 --corr 1000.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 10.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 100.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.001 --corr 1000.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 10.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 100.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.01 --corr 1000.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 10.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 100.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 0.1 --corr 1000.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 10.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 100.0 &
sleep 5
./train.sh --gpu a5000 --mode multi --ortho 1.0 --corr 1000.0 & 

wait


