#!/bin/bash

# Basic configuration
batch_size=128
epochs=300
optimizer=adamw
scheduler=cosine

sbatch run_training.sh $batch_size $epochs $optimizer $scheduler

# varying the batch size
for bs in 32 64 128 256 512 1024; do
    sbatch run_training.sh $bs $epochs $optimizer $scheduler
done

# varying the batch size
for ep in 30 60 90 120 150 180; do
    sbatch run_training.sh $batch_size $ep $optimizer $scheduler
done


# varying the optimizer
for opt in adamw adam sgd; do
    sbatch run_training.sh $batch_size $epochs $opt $scheduler
done

# varying the scheduler
for sch in cosine step multistep none; do
    sbatch run_training.sh $batch_size $epochs $optimizer $sch
done

