#! /bin/bash

python citation.py --dataset cora --epochs 1000 --patience 400 --seed 25190 --batch 32 --dropout 0.5 --nlayer 2 --hid 128 --lr 0.05  --omega 1.15 --rho 0.06 --tau 1.7 --epsilon 0.02 --weight_decay 0.0005

python citation.py --dataset citeseer --epochs 1000 --patience 400 --seed 25190 --batch 32 --dropout 0.1 --nlayer 2 --hid 128 --lr 0.05 --omega 1.1 --rho 0.04 --tau 1.2 --epsilon 0.03 --weight_decay 0.001

python citation.py --dataset pubmed --epochs 1000 --patience 400 --seed 25190 --batch 64 --dropout 0.5 --nlayer 3 --hid 1024 --lr 0.05 --omega 1.15 --rho 0.1 --tau 1.9 --epsilon 0.01 --weight_decay 0.005


python multiclass.py --dataset ogbnarxiv --epochs 1000 --patience 400 --seed 25190 --batch 1024 --dropout 0.5 --nlayer 2 --hid 2048 --lr 0.005 --weight_decay 0.005 --omega 9 --rho 0.85 --tau 1.5 --epsilon 0.008

python multiclass.py --dataset reddit --epochs 2000 --patience 400  --seed 25190 --batch 1024 --dropout 0.2 --nlayer 3 --hid 1024 --lr 0.005 --weight_decay 0.0005 --omega 4.0 --rho 1.1 --tau 0.8 --epsilon 0.008

python multilabel.py --dataset amazon --epochs 1000 --patience 100 --seed 25190 --batch 100000 --dropout 0.4 --nlayer 4 --hid 512 --lr 0.005 --weight_decay 0 --omega 0.9 --rho 1.15 --tau 1.5 --epsilon 0.05

## abla
python multilabel.py --dataset amazon --epochs 1000 --patience 100 --seed 25190 --batch 100000 --dropout 0.4 --nlayer 4 --hid 512 --lr 0.005 --weight_decay 0 --omega 0.3 --rho 1 --tau 1.2 --epsilon 0.08

# idx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
python papers100M.py --dataset papers100M --epochs 1000 --patience 400 --seed 25190 --batch 10000 --dropout 0.1 --nlayer 2 --hid 512 --lr 0.01 --weight_decay 5e-4 --omega 7.4 --rho 1.3 --tau 0.6 --epsilon 0.01 --idx 0
