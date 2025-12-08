#!/bin/bash
cwd=$(pwd)

cd ../../

# launch python venv
# source ../.venv/Scripts/activate
# export python path for local modules
export PYTHONPATH=.

maxinputlinks=13

if [ ! -d "scenarios/cbsstudy/results" ]; then
    mkdir -p "scenarios/cbsstudy/results"
fi

run() {
    maxstages=15
    # $1 the subprocess number
    # $2 the numer of runs to perform
    for i in $(seq 1 $maxstages)
    do
        python3 scenarios/cbsstudy/cbsstudy_run_scenario.py $1 $i True > "scenarios/cbsstudy/results/cbsstudy_individualDelays_il_"$1"_s_"$i"_withCT.test"
        python3 scenarios/cbsstudy/cbsstudy_run_scenario.py $1 $i False > "scenarios/cbsstudy/results/cbsstudy_individualDelays_il_"$1"_s_"$i".test"
    done
}

# run 10 parallel processes
for i in $(seq 2 $maxinputlinks)
do
    run $i &
done
# wait for all background processes to finish
wait

cd $cwd
