#!/bin/bash

source ~/.virtualenvs/tensorflow/bin/activate

rm *.pb

python generate_simple_graph.py
python generate_simple_mlp.py
python generate_simple_mlp_float64.py
python generate_simple_mlp_with_transferable_weights.py
python generate_simple_mlp_with_transferable_weights2.py