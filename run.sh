#!/bin/bash

#################################################
# Script to run all the experiments in sequence #
#################################################

# set env
source myenv/bin/activate

# set pythonpath
export PYTHONPATH=$(pwd)

# run toy_problem 
python experiments/toy1o/main.py  -m optimization=samgs

# run toy_2optimal 
python experiments/toy2o/main.py  -m optimization=samgs

# run celeba 
python experiments/celeba/trainer.py  -m optimization=samgs


# run cityscapes 
python experiments/cityscapes/trainer.py  -m optimization=samgs 

# run NYU v2
python experiments/nyuv2/main.py  -m optimization=samgs

# Wait for all background processes to finish
wait
