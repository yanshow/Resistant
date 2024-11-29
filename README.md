# Resistant

This repository is for source code of "Resistant: Desensitizing Graph Neural Networks Against Adversarial Attacks".

## Environment

- python == 3.8.8
- pytorch == 1.8.2--cuda11.1
- scipy == 1.6.2
- numpy == 1.24.4
- torch_geometric == 2.0.0
- matplotlib == 3.7.5
- optuna == 3.6.1
- deeprobust

You can install library by requirements.txt.
## Perturbed Datasets
You need to install Deeprobust to prepare the perturbed dataset. 

```python
pip install deeprobust
```
Then, you can generate the perturbed graphs via
```python
python generate_attack.py --dataset citeseer --ptb_rate 0.2
```

## Exploratory Experiments
To demonstrate the property that injecting adversarial edges with higher influence results in more severe attacks, we conducted exploratory experiments to evaluate the performance degradation caused by adversarial edges with varying levels of influence.
```python
python drawPropertyHeatmap.py
```
## Main Method
To run Resistant with certain args under poisoning setting, for example:
```python
python main.py --dataset citeseer --ptb_rate 0.20 --k 4 --jt 0.03 --cos 0.1 --alpha 0.55 --lamb 2 --temperature 0.2 --pretrain_ep 180 --pretrain_nc 80 --n_layer 1 --lr 0.008 --epoch 500
```

## Hyper-parameters
We use the optuna library for automatic hyperparameter tuning based on the perturbation rate. 
First, you should install optuna.
```python
pip install optuna
```
Then,you can run the tuning script with the following command:
```python
python train_optuna.py
```
The specific values of Hyper-parameters are listed in Hypermaramter.bat ,which achieve the peak performance against MetaAttack in our experiments.
```python
Hypermaramter.bat
```

