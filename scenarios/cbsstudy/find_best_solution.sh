#!/bin/bash
cd ..

# launch python venv
source ../.venv/Scripts/activate
# export python path for local modules
export PYTHONPATH=.

# start X parallel processes that run python3 optimization/optimization/demo_call.py and pipe the output to a result file

run() {
    # $1 the subprocess number
    # $2 the numer of runs to perform
    for i in $(seq 1 $2)
    do
        python3 cbsstudy/cbsstudy_run_scenario.py > cbsstudy/results/result_$1_$i.txt 
    done
}

# create the results dir if it does not exist
mkdir -p results

# run 10 parallel processes
for i in {1..10}
do
    run $i 10 &
done
# wait for all background processes to finish
wait
