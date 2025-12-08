#!/bin/bash
cwd=$(pwd)

cd ../../

# launch python venv
source ../venv/Scripts/activate
# export python path for local modules
export PYTHONPATH=.

if [ ! -d "scenarios/carnet/results" ]; then
    mkdir -p "scenarios/carnet/results"
fi

python3 scenarios/carnet/car_run_scenario.py > "scenarios/carnet/results/car_study.txt"

cd $cwd
