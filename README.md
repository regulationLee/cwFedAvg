# cwFedAvg
ICCV 2025 accepted paper, Class-Wise Federated Averaging for Efficient Personalization ([arXiv]([https://arxiv.org/abs/2406.07800]))


### Dataset setting
```sh
cd dataset
python generate_cifar10.py noniid - dir 20 0.1 # noniid + class imbalance, practical setting(0.1), 20 clients 
```



### PFL training using cwFedAvg algorithm
```sh
# cwFedAvg with decision layer only
cd system
python main.py -data Cifar10_dir_N_20_alpha_0.1 -m cnn -algo cwFedAvg -cw -wdr -wd 10 -plt -ncw 1 -go test

# cwFedAvg with whole layer
cd system
python main.py -data Cifar10_dir_N_20_alpha_0.1 -m cnn -algo cwFedAvg -cw -wdr -wd 10 -go test
