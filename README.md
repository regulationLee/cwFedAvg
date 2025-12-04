# Class-Wise Federated Averaging for Efficient Personalization
Code for the paper "Class-Wise Federated Averaging for Efficient Personalization" a.k.a cwFedAvg, accepted to ICCV 2025 ([Main](https://iccv.thecvf.com/virtual/2025/poster/2173)).

Federated learning (FL) enables collaborative model training across distributed clients without centralizing data.
However, existing approaches such as Federated Averaging (FedAvg) often perform poorly with heterogeneous data distributions, failing to achieve personalization owing to their inability to capture class-specific information effectively. We propose Class-wise Federated Averaging (cwFedAvg), a novel personalized FL(PFL) framework that performs Federated Averaging for each class, to overcome the personalization limitations of FedAvg. cwFedAvg creates class-specific global models via weighted aggregation of local models using class distributions, and subsequently combines them to generate personalized local models. We further propose Weight Distribution Regularizer (WDR), which encourages deep networks to encode class-specific information efficiently by aligning empirical and approximated class distributions derived from output layer weights, to facilitate effective class-wise aggregation. Our experiments demonstrate the superior
performance of cwFedAvg with WDR over existing PFL methods through efficient personalization while maintaining the communication cost of FedAvg and avoiding additional local training and pairwise computations. (Paper: [arXiv](https://arxiv.org/abs/2406.07800) | Video: [Youtube](https://www.youtube.com/watch?v=z8LwxOio1Qs))



## ⚙️ Implementation
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
```


## 📌 Framework Foundation
This repository contains the implementation code for our paper, built upon **[PFLlib: Personalized Federated Learning Library and Benchmark](https://github.com/TsingZ0/PFLlib)**.

PFLlib is a comprehensive Python library featuring 39+ federated learning algorithms, 24 datasets, and specialized support for addressing data heterogeneity in personalized federated learning scenarios.

## 🙏 Acknowledgments
We gratefully acknowledge the PFLlib team for providing an excellent foundation for personalized federated learning research.
