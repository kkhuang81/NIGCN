#/bash/bin

# --no-opt  //disable optimization

# type = {0, 1, 2, 3}

python citation.py --dataset cora --epochs 1000 --patience 400 --rr 0.5 --seed 25190 --level 18 --alpha 0.05 --dropout 0.5 --batch 32 --hid 128 --lr 0.01 --epsilon 0.02 --type $type  

python citation.py --dataset citeseer --batch 32 --dropout 0.1 --epochs 1000 --hid 128 --lr 0.01 --patience 400 --rr 0.5 --epsilon 0.03 --seed 25190 --alpha 0.05 --level 16 --nlayer 2 --type $type  

python citation.py --dataset pubmed --alpha 0.1 --batch 64 --dropout 0.5 --epochs 1000 --hid 1024 --level 21 --lr 0.01 --patience 400 --rr 0.5 --epsilon 0.01 --seed 25190 --nlayer 3 --type $type 

python multiclass.py --dataset ogbnarxiv --epochs 1000 --epsilon 0.008 --patience 400 --rr 0.5 --seed 25190 --weight_decay 0 --dropout 0.5 --batch 1024 --nlayer 2 --hid 2048 --lr 0.005 --level 10 --lamb 7 --type $type  

python multiclass.py --dataset reddit --batch 1024 --rr 0.5 --weight_decay 0 --seed 25190 --level 7 --dropout 0.2 --lamb 6 --lr 0.005 --nlayer 3 --epochs 2000 --hid 1024 --dev 1 --patience 400 --epsilon 0.008 --type $type 

python multilabel.py --dataset amazon --batch 100000 --epsilon 0.04 --rr 0.5 --weight_decay 0 --seed 25190 --level 4 --dropout 0.4 --alpha 0.7 --lr 0.01 --nlayer 4 --epochs 1000 --hid 512 --dev 1 --patience 100 --type $type

# 0.08 and --bia='bn' for GTRO in Ablation Study
python multilabel.py --dataset amazon --batch 100000 --epsilon 0.08 --rr 0.5 --weight_decay 0 --seed 25190 --level 4 --dropout 0.4 --alpha 0.7 --lr 0.01 --nlayer 4 --epochs 1000 --hid 512 --bia='bn' --dev 1 --patience 100 --type $type

# idx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
python papers100M.py --dataset papers100M --batch 10000 --rr 0.5 --seed 25190 --dropout 0.1 --lamb 7 --hid 512 --dev 1 --patience 400 --weight_decay 5e-4 --level 5 --lr 0.01 --nlayers 2 --epsilon 0.01 --type $type --idx $idx
