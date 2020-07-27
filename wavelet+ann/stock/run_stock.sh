#!/bin/bash


echo "model_3: ann"
echo "ahead_step: 1"
python ./stock/stock_data.py --ahead_step=1 --time_window=5 --num_frequencies=5 --lstm_units=8 --max_training_iters=50 --keep_prob=1.0 --model_structure=3 --notes=ann --learning_rate=0.01


echo "done"
