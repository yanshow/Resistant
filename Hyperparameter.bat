python main.py --dataset citeseer --ptb_rate 0.00 --k 1 --jt 0.01 --cos 0.0 --alpha 0.3 --lamb 2.15 --temperature 0.05 --pretrain_ep 120 --pretrain_nc 60 --n_layer 1 --lr 0.009 --epoch 250
python main.py --dataset citeseer --ptb_rate 0.05 --k 1 --jt 0.01 --cos 0.2 --alpha 0.8 --lamb 1.3 --temperature 0.05 --pretrain_ep 140 --pretrain_nc 150 --n_layer 1 --lr 0.004 --epoch 450
python main.py --dataset citeseer --ptb_rate 0.10 --k 1 --jt 0.03 --cos 0.1 --alpha 0.85 --lamb 2.9 --temperature 0.6 --pretrain_ep 250 --pretrain_nc 50 --n_layer 1 --lr 0.01 --epoch 400
python main.py --dataset citeseer --ptb_rate 0.15 --k 3 --jt 0.01 --cos 0.1 --alpha 0.95 --lamb 3.0 --temperature 0.45 --pretrain_ep 240 --pretrain_nc 60 --n_layer 1 --lr 0.005 --epoch 200
python main.py --dataset citeseer --ptb_rate 0.20 --k 4 --jt 0.03 --cos 0.1 --alpha 0.55 --lamb 2 --temperature 0.2 --pretrain_ep 180 --pretrain_nc 80 --n_layer 1 --lr 0.008 --epoch 500
python main.py --dataset cora --ptb_rate 0 --k 7 --jt 0 --cos 0.1 --alpha 0.2 --lamb 1.7 --temperature 0 --pretrain_ep 220 --pretrain_nc 240 --n_layer 1 --lr 0.006 --epoch 350
python main.py --dataset cora --ptb_rate 0.05 --k 2 --jt 0.03 --cos 0.1 --alpha 0.7 --lamb 0.85 --temperature 0.6 --pretrain_ep 240 --pretrain_nc 170 --n_layer 1 --lr 0.01 --epoch 300
python main.py --dataset cora --ptb_rate 0.10 --k 8 --jt 0.03 --cos 0.1 --alpha 0.05 --lamb 2.55 --temperature 0.7 --pretrain_ep 100 --pretrain_nc 230 --n_layer 2 --lr 0.009 --epoch 500
python main.py --dataset cora --ptb_rate 0.15 --k 7 --jt 0.03 --cos 0.1 --alpha 0.95 --lamb 2.7 --temperature 0.2 --pretrain_ep 10 --pretrain_nc 90 --n_layer 1 --lr 0.006 --epoch 500
python main.py --dataset cora --ptb_rate 0.20 --k 9 --jt 0.05 --cos 0.1 --alpha 0.15 --lamb 1.4 --temperature 0.2 --pretrain_ep 20 --pretrain_nc 50 --n_layer 2 --lr 0.01 --epoch 450
python main.py --dataset polblogs --ptb_rate 0.00 --k 0 --jt 0.03 --cos 0.1 --alpha 0.6 --lamb 0.95 --temperature 0.6 --pretrain_ep 110 --pretrain_nc 230 --n_layer 1 --lr 0.005 --epoch 300
python main.py --dataset polblogs --ptb_rate 0.05 --k 15 --jt 0.02 --cos 0.2 --alpha 0.85 --lamb 0.1 --temperature 0.1 --pretrain_ep 80 --pretrain_nc 20 --n_layer 3 --lr 0.009 --epoch 200
python main.py --dataset polblogs --ptb_rate 0.1 --k 15 --jt 0.02 --cos 0.2 --alpha 0.85 --lamb 0.1 --temperature 0.1 --pretrain_ep 80 --pretrain_nc 20 --n_layer 3 --lr 0.009 --epoch 200
python main.py --dataset polblogs --ptb_rate 0.15 --k 14 --jt 0.02 --cos 0.1 --alpha 0.95 --lamb 0.1 --temperature 0.45 --pretrain_ep 50 --pretrain_nc 10 --n_layer 3 --lr 0.007 --epoch 250
python main.py --dataset polblogs --ptb_rate 0.2 --k 18 --jt 0.01 --cos 0.3 --alpha 0.4 --lamb 0 --temperature 0.45 --pretrain_ep 80 --pretrain_nc 10 --n_layer 3 --lr 0.004 --epoch 500
python main.py --dataset cora_ml --ptb_rate 0 --k 6 --jt 0 --cos 0.1 --temperature 0.95 --alpha 0.05 --lamb 2.7 --lr 0.01 --n_layer 1 --epoch 250 --pretrain_ep 50 --pretrain_nc 250
python main.py --dataset cora_ml --ptb_rate 0.05 --k 3 --jt 0.01 --cos 0.2 --temperature 0.3 --alpha 0.8 --lamb 2.45 --lr 0.01 --n_layer 1 --epoch 250 --pretrain_ep 230 --pretrain_nc 140
python main.py --dataset cora_ml --ptb_rate 0.1 --k 4 --jt 0.01 --cos 0.3 --temperature 0.1 --alpha 0.9 --lamb 2.0 --lr 0.008 --n_layer 1 --epoch 400 --pretrain_ep 160 --pretrain_nc 100
python main.py --dataset cora_ml --ptb_rate 0.15 --k 4 --jt 0.05 --cos 0.3 --temperature 0.3 --alpha 0.95 --lamb 2.75 --lr 0.01 --n_layer 1 --epoch 500 --pretrain_ep 210 --pretrain_nc 180
python main.py --dataset cora_ml --ptb_rate 0.2 --k 2 --jt 0.05 --cos 0.2 --temperature 0.5 --alpha 0.95 --lamb 1.1 --lr 0.007 --n_layer 1 --epoch 500 --pretrain_ep 40 --pretrain_nc 120


