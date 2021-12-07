#!/bin/sh
# for Cat&Dog use --loss=ce for cross-entropy loss
python binary_auc.py --loss=auc --dataset=2  --keep_index=1.0 &&
python binary_auc.py --loss=auc --dataset=2  --keep_index=0.4 &&
python binary_auc.py --loss=auc --dataset=2  --keep_index=0.2 &&
python binary_auc.py --loss=auc --dataset=2  --keep_index=0.1 &&

# for CIFAR10
python binary_auc.py --loss=auc --dataset=10  --keep_index=1.0 &&
python binary_auc.py --loss=auc --dataset=10  --keep_index=0.4 &&
python binary_auc.py --loss=auc --dataset=10  --keep_index=0.2 &&
python binary_auc.py --loss=auc --dataset=10  --keep_index=0.1 &&

# for CIFAR100
python binary_auc.py --loss=auc --dataset=100 --split_index=49 --keep_index=1.0 &&
python binary_auc.py --loss=auc --dataset=100 --split_index=49 --keep_index=0.4 &&
python binary_auc.py --loss=auc --dataset=100 --split_index=49 --keep_index=0.2 &&
python binary_auc.py --loss=auc --dataset=100 --split_index=49 --keep_index=0.1 &&

# for STl10
python binary_auc.py --loss=auc --dataset=10 --is_stil10=True --keep_index=1.0 &&
python binary_auc.py --loss=auc --dataset=10 --is_stil10=True --keep_index=0.4 &&
python binary_auc.py --loss=auc --dataset=10 --is_stil10=True --keep_index=0.2 &&
python binary_auc.py --loss=auc --dataset=10 --is_stil10=True --keep_index=0.1