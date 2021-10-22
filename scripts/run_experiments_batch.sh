#!/bin/bash
screen -Smd up-agnews ./run_experiments.sh 0 agnews
screen -Smd up-imdb ./run_experiments.sh 0 imdb
screen -Smd up-sst2 ./run_experiments.sh 0 sst2
screen -Smd up-yahoo ./run_experiments.sh 0 yahoo
screen -Smd up-yelp ./run_experiments.sh 0 yelp
screen -Smd up-rte ./run_experiments.sh 0 rte
screen -Smd up-boolq ./run_experiments.sh 0 boolq
