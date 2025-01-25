#!/bin/bash

# Basic configuration
batch_size=128
epochs=300
optimizer=adamw
scheduler=cosine

sbatch run_training.sh $batch_size $epochs $optimizer $scheduler

# varying the number of epochs
sbatch run_training.sh batch_size 30 adamw cosine
sbatch run_training.sh batch_size 60 adamw cosine
sbatch run_training.sh batch_size 90 adamw cosine
sbatch run_training.sh batch_size 120 adamw cosine
sbatch run_training.sh batch_size 150 adamw cosine

# varying the batch size
sbatch run_training.sh 32 epochs adamw cosine
sbatch run_training.sh 64 epochs adamw cosine
sbatch run_training.sh 256 epochs adamw cosine
sbatch run_training.sh 512 epochs adamw cosine
sbatch run_training.sh 1024 epochs adamw cosine

# varying the optimizer
sbatch run_training.sh batch_size epochs sgd cosine
sbatch run_training.sh batch_size epochs adam cosine
sbatch run_training.sh batch_size epochs adamw cosine

# varying the scheduler
sbatch run_training.sh batch_size epochs optimizer cosine
sbatch run_training.sh batch_size epochs optimizer step
sbatch run_training.sh batch_size epochs optimizer multistep
sbatch run_training.sh batch_size epochs optimizer none

