#!/bin/bash
echo "Independent training of multiple stages recommendation models on dataset $1"
cd ../

for model in $@; do
    if [[ $model == $1 ]]; then
        continue
    fi
    echo "begin training $model on $1 dataset..."
    python warmup.py -d $1 -m $model
done
