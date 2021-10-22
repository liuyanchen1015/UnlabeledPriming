#!/bin/bash
screen -dmS up-agnews sh -c './run_experiments.sh 0 agnews; exec bash'
screen -dmS up-imdb sh -c './run_experiments.sh 0 imdb; exec bash'
screen -dmS up-sst2 sh -c './run_experiments.sh 0 sst2; exec bash'
screen -dmS up-yahoo sh -c './run_experiments.sh 0 yahoo; exec bash'
screen -dmS up-yelp sh -c './run_experiments.sh 0 yelp; exec bash'
screen -dmS up-rte sh -c './run_experiments.sh 0 rte; exec bash'
screen -dmS up-boolq sh -c './run_experiments.sh 0 boolq; exec bash'
