#!/bin/bash 

sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 1.0 --gsep -0.001 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 1.0 --isep -0.001 --il1 0.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 0.1 --gsep -0.001 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 0.1 --isep -0.001 --il1 0.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 0.01 --gsep -0.001 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 0.01 --isep -0.001 --il1 0.0 & 

sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 1.0 --gsep -0.01 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 1.0 --isep -0.01 --il1 0.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 0.1 --gsep -0.01 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 0.1 --isep -0.01 --il1 0.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 0.01 --gsep -0.01 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 0.01 --isep -0.01 --il1 0.0 & 

sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 1.0 --gsep -0.1 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 1.0 --isep -0.1 --il1 0.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 0.1 --gsep -0.1 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 0.1 --isep -0.1 --il1 0.0 & 
sleep 5
./train.sh --gpu a5000 --mode multi --corr 10.0 \
  --gortho 0.0 --gcrs_ent 20.0 --gclst 0.01 --gsep -0.1 --gl1 0.0 \
  --iortho 0.01 --icrs_ent 20.0 --iclst 0.01 --isep -0.1 --il1 0.0 & 

wait


