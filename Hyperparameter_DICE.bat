python main.py --dataset cora --ptb_rate 0.05 --k 6 --jt 0.0 --cos 0.1 --alpha 0.6 --lamb 2.05 --temperature 0.9 --pretrain_ep 50 --pretrain_nc 170 --n_layer 1 --lr 0.01 --epoch 450 --attack DICE
python main.py --dataset cora --ptb_rate 0.1 --k 7 --jt 0.0 --cos 0.0 --alpha 0.3 --lamb 1.95 --temperature 0.9 --pretrain_ep 80 --pretrain_nc 220 --n_layer 1 --lr 0.005 --epoch 200 --attack DICE
python main.py --dataset cora --ptb_rate 0.15 --k 6 --jt 0.0 --cos 0.0 --alpha 0.8 --lamb 2.5 --temperature 0.8 --pretrain_ep 140 --pretrain_nc 150 --n_layer 2 --lr 0.003 --epoch 250 --attack DICE
python main.py --dataset cora --ptb_rate 0.20 --k 6 --jt 0.02 --cos 0.0 --alpha 0.3 --lamb 2.05 --temperature 0.0 --pretrain_ep 250 --pretrain_nc 40 --n_layer 1 --lr 0.004 --epoch 450 --attack DICE

python main.py --dataset citeseer --ptb_rate 0.05 --k 9 --jt 0.03 --cos 0.2 --alpha 0.85 --lamb 0.6 --temperature 0.2 --pretrain_ep 220 --pretrain_nc 220 --n_layer 1 --lr 0.003 --epoch 500 --attack DICE
python main.py --dataset citeseer --ptb_rate 0.1 --k 1 --jt 0.05 --cos 0.0 --alpha 0.4 --lamb 0.95 --temperature 0.25 --pretrain_ep 240 --pretrain_nc 110 --n_layer 1 --lr 0.009 --epoch 200 --attack DICE
python main.py --dataset citeseer --ptb_rate 0.15 --k 0 --jt 0.04 --cos 0.1 --alpha 0.8 --lamb 1.95 --temperature 0.15 --pretrain_ep 110 --pretrain_nc 90 --n_layer 1 --lr 0.005 --epoch 400 --attack DICE
python main.py --dataset citeseer --ptb_rate 0.2 --k 3 --jt 0.02 --cos 0.0 --alpha 0.7 --lamb 2.2 --temperature 0.1 --pretrain_ep 120 --pretrain_nc 90 --n_layer 1 --lr 0.003 --epoch 300 --attack DICE

python main.py --dataset polblogs --ptb_rate 0.05 --k 0 --jt 0.04 --cos 0.1 --alpha 0.7 --lamb 2.05 --temperature 0.1 --pretrain_ep 130 --pretrain_nc 150 --n_layer 3 --lr 0.001 --epoch 250 --attack DICE
python main.py --dataset polblogs --ptb_rate 0.1 --k 0 --jt 0.03 --cos 0.2 --alpha 0.65 --lamb 2.6 --temperature 0.7 --pretrain_ep 20 --pretrain_nc 40 --n_layer 2 --lr 0.009 --epoch 350 --attack DICE
python main.py --dataset polblogs --ptb_rate 0.15 --k 0 --jt 0.05 --cos 0.2 --alpha 0.95 --lamb 2.6 --temperature 0.95 --pretrain_ep 220 --pretrain_nc 110 --n_layer 2 --lr 0.009 --epoch 300 --attack DICE
python main.py --dataset polblogs --ptb_rate 0.2 --k 2 --jt 0.05 --cos 0.1 --alpha 0.95 --lamb 2.2 --temperature 0.15 --pretrain_ep 120 --pretrain_nc 60 --n_layer 2 --lr 0.009 --epoch 450 --attack DICE

python main.py --dataset cora_ml --ptb_rate 0.05 --k 1 --jt 0.0 --cos 0.0 --alpha 0.5 --lamb 0.5 --temperature 0.75 --pretrain_ep 200 --pretrain_nc 250 --n_layer 1 --lr 0.009 --epoch 350 --attack DICE
python main.py --dataset cora_ml --ptb_rate 0.1 --k 0 --jt 0.0 --cos 0.0 --alpha 0.5 --lamb 2.7 --temperature 0.1 --pretrain_ep 150 --pretrain_nc 210 --n_layer 1 --lr 0.01 --epoch 400 --attack DICE
python main.py --dataset cora_ml --ptb_rate 0.15 --k 6 --jt 0.0 --cos 0.0 --alpha 0.5 --lamb 2.7 --temperature 0.6 --pretrain_ep 210 --pretrain_nc 200 --n_layer 1 --lr 0.009 --epoch 200 --attack DICE
python main.py --dataset cora_ml --ptb_rate 0.2 --k 3 --jt 0.04 --cos 0.1 --alpha 0.55 --lamb 1.8 --temperature 0.05 --pretrain_ep 170 --pretrain_nc 210 --n_layer 1 --lr 0.01 --epoch 500 --attack DICE