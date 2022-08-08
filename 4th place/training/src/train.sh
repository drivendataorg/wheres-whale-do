#!/bin/bash
N_FOLDS=5
for ((fold_id=0;fold_id<N_FOLDS;fold_id++)); do
    python src/pretrain_256.py --data_dir $1 --model_dir $2 --fold $fold_id
done

for ((fold_id=0;fold_id<N_FOLDS;fold_id++)); do
    python src/train_256.py --data_dir $1 --model_dir $2 --fold $fold_id
done

for ((fold_id=0;fold_id<N_FOLDS;fold_id++)); do
    python src/pretrain_416.py --data_dir $1 --model_dir $2 --fold $fold_id
done

for ((fold_id=0;fold_id<N_FOLDS;fold_id++)); do
    python src/train_416.py --data_dir $1 --model_dir $2 --fold $fold_id
done

for ((fold_id=0;fold_id<N_FOLDS;fold_id++)); do
    python src/pretrain_448.py --data_dir $1 --model_dir $2 --fold $fold_id
done

for ((fold_id=0;fold_id<N_FOLDS;fold_id++)); do
    python src/train_448.py --data_dir $1 --model_dir $2 --fold $fold_id
done