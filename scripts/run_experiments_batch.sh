#!/bin/bash
screen -dmS up-agnews sh -c './run_experiments.sh 0 agnews; exec bash'
screen -dmS up-imdb sh -c './run_experiments.sh 1 imdb; exec bash'
screen -dmS up-yahoo sh -c './run_experiments.sh 2 yahoo; exec bash'
screen -dmS up-yelp sh -c './run_experiments.sh 3 yelp; exec bash'
