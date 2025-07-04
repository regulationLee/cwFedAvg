import copy
import torch
import platform
import argparse
import os
import json
import time
import warnings
import numpy as np
import torchvision
import logging

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

vocab_size = 98635  # 98635 for AG_News and 399198 for Sogou_News
max_len = 200
emb_dim = 32


def load_select_modes(args, model_str):
    # Generate args.model
    if model_str == "mlr":  # convex
        if "mnist" in args.dataset:
            args.model = Mclr_Logistic(1 * 28 * 28, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = Mclr_Logistic(3 * 32 * 32, num_classes=args.num_classes).to(args.device)
        else:
            args.model = Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

    elif model_str == "cnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "FashionMNIST" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
        elif "omniglot" in args.dataset:
            args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
        elif "Digit5" in args.dataset:
            args.model = Digit5CNN().to(args.device)
        else:
            args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

    elif model_str == "dnn":  # non-convex
        if "mnist" in args.dataset:
            args.model = DNN(1 * 28 * 28, 100, num_classes=args.num_classes).to(args.device)
        elif "Cifar10" in args.dataset:
            args.model = DNN(3 * 32 * 32, 100, num_classes=args.num_classes).to(args.device)
        else:
            args.model = DNN(60, 20, num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet":
        args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)

        # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)

    elif model_str == "resnet8":
        args.model = resnet8(num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet10":
        args.model = resnet10(num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet18":
        args.model = resnet18(num_classes=args.num_classes).to(args.device)

    elif model_str == "resnet34":
        args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "alexnet":
        args.model = alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)

        # args.model = alexnet(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "googlenet":
        args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False,
                                                  num_classes=args.num_classes).to(args.device)

        # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "mobilenet_v2":
        args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)

        # args.model = mobilenet_v2(pretrained=True).to(args.device)
        # feature_dim = list(args.model.fc.parameters())[0].shape[1]
        # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

    elif model_str == "shufflenet":
        args.model = torchvision.models.shufflenet_v2_x1_0(pretrained=False, num_classes=args.num_classes).to(args.device)

    elif model_str == "lstm":
        args.model = LSTMNet(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
            args.device)

    elif model_str == "bilstm":
        args.model = BiLSTM_TextClassification(input_size=vocab_size, hidden_size=emb_dim,
                                               output_size=args.num_classes,
                                               num_layers=1, embedding_dropout=0, lstm_dropout=0,
                                               attention_dropout=0,
                                               embedding_length=emb_dim).to(args.device)

    elif model_str == "fastText":
        args.model = fastText(hidden_dim=emb_dim, vocab_size=vocab_size, num_classes=args.num_classes).to(
            args.device)

    elif model_str == "TextCNN":
        args.model = TextCNN(hidden_dim=emb_dim, max_len=max_len, vocab_size=vocab_size,
                             num_classes=args.num_classes).to(args.device)

    elif model_str == "Transformer":
        args.model = TransformerModel(ntoken=vocab_size, d_model=emb_dim, nhead=8, d_hid=emb_dim, nlayers=2,
                                      num_classes=args.num_classes).to(args.device)

    elif model_str == "AmazonMLP":
        args.model = AmazonMLP().to(args.device)

    elif model_str == "harcnn":
        if args.dataset == 'har':
            args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                pool_kernel_size=(1, 2)).to(args.device)
        elif args.dataset == 'pamap':
            args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9),
                                pool_kernel_size=(1, 2)).to(args.device)

    elif model_str == "Homo":
        args.model = ['resnet8(num_classes=args.num_classes)']

    elif model_str == "Ht0":
        args.model = [
            'resnet8(num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.shufflenet_v2_x1_0(pretrained=False)',
            'torchvision.models.efficientnet_b0(pretrained=False)',
        ]

    elif model_str == "HtFE2":
        args.model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
        ]

    elif model_str == "HtFE3":
        args.model = [
            'resnet10(num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
        ]

    elif model_str == "HtFE4":
        args.model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)'
        ]

    elif model_str == "HtFE8":
        args.model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
            # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
        ]

    elif model_str == "HtFE9":
        args.model = [
            'resnet4(num_classes=args.num_classes)',
            'resnet6(num_classes=args.num_classes)',
            'resnet8(num_classes=args.num_classes)',
            'resnet10(num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
        ]

    elif model_str == "HtFE8-HtC4":
        args.model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)'
        ]
        args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'
        args.heads = [
            'Head(hidden_dims=[512], num_classes=args.num_classes)',
            'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
            'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
            'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
        ]

    elif model_str == "Res34-HtC4":
        args.model = [
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
        ]
        args.global_model = 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)'
        args.heads = [
            'Head(hidden_dims=[512], num_classes=args.num_classes)',
            'Head(hidden_dims=[512, 512], num_classes=args.num_classes)',
            'Head(hidden_dims=[512, 256], num_classes=args.num_classes)',
            'Head(hidden_dims=[512, 128], num_classes=args.num_classes)',
        ]

    elif model_str == "HCNNs8":
        args.model = [
            'CNN(num_cov=1, hidden_dims=[], in_features=1, num_classes=args.num_classes)',
            'CNN(num_cov=2, hidden_dims=[], in_features=1, num_classes=args.num_classes)',
            'CNN(num_cov=1, hidden_dims=[512], in_features=1, num_classes=args.num_classes)',
            'CNN(num_cov=2, hidden_dims=[512], in_features=1, num_classes=args.num_classes)',
            'CNN(num_cov=1, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)',
            'CNN(num_cov=2, hidden_dims=[1024], in_features=1, num_classes=args.num_classes)',
            'CNN(num_cov=1, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)',
            'CNN(num_cov=2, hidden_dims=[1024, 512], in_features=1, num_classes=args.num_classes)',
        ]

    elif model_str == "ViTs":
        args.model = [
            'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)',
            'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)',
            'torchvision.models.vit_l_16(image_size=32, num_classes=args.num_classes)',
            'torchvision.models.vit_l_32(image_size=32, num_classes=args.num_classes)',
        ]

    elif model_str == "HtM10":
        args.model = [
            'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
            'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
            'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet101(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.resnet152(pretrained=False, num_classes=args.num_classes)',
            'torchvision.models.vit_b_16(image_size=32, num_classes=args.num_classes)',
            'torchvision.models.vit_b_32(image_size=32, num_classes=args.num_classes)'
        ]

    elif model_str == "NLP_all":
        args.model = [
            'fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, output_size=args.num_classes, num_layers=1, embedding_dropout=0, lstm_dropout=0, attention_dropout=0, embedding_length=args.feature_dim)',
            'TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, num_classes=args.num_classes)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)'
        ]

    elif model_str == "NLP_Transformers-nhead=8":
        args.model = [
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
        ]

    elif model_str == "NLP_Transformers-nlayers=4":
        args.model = [
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
        ]

    elif model_str == "NLP_Transformers":
        args.model = [
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=1, nlayers=1, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=2, nlayers=2, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=4, nlayers=4, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=8, num_classes=args.num_classes, max_len=args.max_len)',
            'TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=16, nlayers=16, num_classes=args.num_classes, max_len=args.max_len)',
        ]

    elif model_str == "MLPs":
        args.model = [
            'AmazonMLP(feature_dim=[])',
            'AmazonMLP(feature_dim=[200])',
            'AmazonMLP(feature_dim=[500])',
            'AmazonMLP(feature_dim=[1000, 500])',
            'AmazonMLP(feature_dim=[1000, 500, 200])',
        ]

    elif model_str == "MLP_1layer":
        args.model = [
            'AmazonMLP(feature_dim=[200])',
            'AmazonMLP(feature_dim=[500])',
        ]

    elif model_str == "MLP_layers":
        args.model = [
            'AmazonMLP(feature_dim=[500])',
            'AmazonMLP(feature_dim=[1000, 500])',
            'AmazonMLP(feature_dim=[1000, 500, 200])',
        ]

    else:
        raise NotImplementedError

    return args.model