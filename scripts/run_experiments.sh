#!/bin/bash
GPU=$1  # the GPU to run the experiments on
TASK=$2 # the task we want to evaluate on

export CUDA_VISIBLE_DEVICES=$GPU
source /mounts/Users/cisintern/schickt/venvs/up/bin/activate

cd "$HOME"/up/final_version || exit

python run_experiment.py -n -t "${TASK}" -k 0 --batch_size 4 -nt 10000 -nu 10000
python run_experiment.py -n -t "${TASK}" -k 3 -c 0 -p uniform --batch_size 4 -nt 10000 -nu 10000
python run_experiment.py -n -t "${TASK}" -k 10 -c 0 -p uniform --batch_size 4 -nt 10000 -nu 10000
python run_experiment.py -n -t "${TASK}" -k 50 -c 0 -p uniform --batch_size 4 -nt 10000 -nu 10000
python run_experiment.py -n -t "${TASK}" -k 3 -c 0 -p concat --batch_size 4 -nt 10000 -nu 10000
