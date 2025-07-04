# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

# !/usr/bin/env python
import copy
from collections import defaultdict

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

from flcore.servers.serveravgCW import cwFedAvg
from flcore.servers.servermd import FedMD
from flcore.servers.serverdf import FedDF
from flcore.servers.servercst import FedCST
from flcore.servers.serverIFCA import IFCA
from flcore.servers.serverCFL import CFL
from flcore.servers.serveravg import FedAvg
# from flcore.servers.serverpFedMe import pFedMe
# from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
# from flcore.servers.servermtl import FedMTL
# from flcore.servers.serverlocal import Local
# from flcore.servers.serverper import FedPer
# from flcore.servers.serverapfl import APFL
# from flcore.servers.serverditto import Ditto
# from flcore.servers.serverrep import FedRep
from flcore.servers.serverah import FedAH
from flcore.servers.serverphp import FedPHP
# from flcore.servers.serverbn import FedBN
# from flcore.servers.serverrod import FedROD
# from flcore.servers.serverproto import FedProto
# from flcore.servers.serverdyn import FedDyn
# from flcore.servers.servermoon import MOON
# from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
# from flcore.servers.servergen import FedGen
# from flcore.servers.serverscaffold import SCAFFOLD
# from flcore.servers.serverdistill import FedDistill
# from flcore.servers.serverala import FedALA
from flcore.servers.serverpac import FedPAC
# from flcore.servers.serverlg import LG_FedAvg
# from flcore.servers.servergc import FedGC
# from flcore.servers.serverfml import FML
# from flcore.servers.serverkd import FedKD
# from flcore.servers.serverpcl import FedPCL
# from flcore.servers.servercp import FedCP
# from flcore.servers.servergpfl import GPFL
# from flcore.servers.serverntd import FedNTD
# from flcore.servers.servergh import FedGH
# from flcore.servers.serveravgDBE import FedAvgDBE

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data, plotting_trial_result
from utils.mem_utils import MemReporter
from load_models import *

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

# hyper-params for Text tasks
vocab_size = 98635  # 98635 for AG_News and 399198 for Sogou_News
max_len = 200
emb_dim = 32


def run(args):
    torch.manual_seed(args.seed)

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        args.model = load_select_modes(args, model_str)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "cwFedAvg":
            if args.partial_layer_train:
                layer_names = []
                for name, _ in args.model.named_children():
                    layer_names.append(name)

                args.layer_groups = {
                    "common": layer_names[:-args.cw_layer_num],
                    "cw": layer_names[-args.cw_layer_num:]
                }

                for key, value in args.layer_groups.items():
                    print(f'{key}: {value}')
            else:
                if args.add_proto:
                    args.model = BaseHeadSplit_prototype(args)
                else:
                    args.head = copy.deepcopy(args.model.fc)
                    args.model.fc = nn.Identity()
                    args.model = BaseHeadSplit(args.model, args.head)

            server = cwFedAvg(args, i)

        elif args.algorithm == "FedMD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedMD(args, i)

        elif args.algorithm == "FedDF":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDF(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)

            server = FedPAC(args, i)

        elif args.algorithm == "FedCST":
            server = FedCST(args, i)

        elif args.algorithm == 'IFCA':
            server = IFCA(args, i)

        elif args.algorithm == 'CFL':
            server = CFL(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedAH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAH(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        # elif args.algorithm == "FedProto":
        #     args.head = copy.deepcopy(args.model.fc)
        #     args.model.fc = nn.Identity()
        #     args.model = BaseHeadSplit(args.model, args.head)
        #     server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FedDistill":
            server = FedDistill(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedAvgDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvgDBE(args, i)

        else:
            raise NotImplementedError

        if isinstance(args.model, list):
            for model in args.model:
                print(model)
        else:
            print(args.model)

        server.train()

        time_list.append(time.time() - start)

        plotting_trial_result(args)

    avg_time = np.average(time_list)
    minutes, seconds = divmod(avg_time, 60)
    hours, minutes = divmod(minutes, 60)
    print(f"\nAverage time cost: {int(hours)}h {int(minutes)}m {round(seconds, 2)}s.")

    # Global average
    # average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    if args.device == 'cuda':
        reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-seed', "--seed", type=int, default=0)

    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=2)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-dp', "--privacy", type=bool, default=False,
                        help="differential privacy")
    parser.add_argument('-dps', "--dp_sigma", type=float, default=0.0)
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    # required
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCST")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")

    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="HtFE2")
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")

    parser.add_argument('-nifca', "--num_clusters", type=int, default=2)
    parser.add_argument('-ftc', "--fine_tune_c", action='store_true')

    # cwFedAvg
    parser.add_argument('-cw', "--add_cw", action='store_true')
    parser.add_argument('-wdr', "--add_wdr", action='store_true')
    parser.add_argument('-dlo', "--decision_layer_only", action='store_true')
    parser.add_argument('-plt', "--partial_layer_train", action='store_true')
    parser.add_argument('-spl', "--split_train", action='store_true')
    parser.add_argument('-hlr', "--head_lr", type=float, default=0.005)
    parser.add_argument('-hbs', "--head_bs", type=int, default=10)
    parser.add_argument('-ncw', "--cw_layer_num", type=int, default=1)

    parser.add_argument('-apr', "--add_proto", action='store_true')

    parser.add_argument('-gt', "--use_true_dist", action='store_true')
    parser.add_argument('-be', "--batch_eval", action='store_true')
    parser.add_argument('-clw', "--clip_weight", action='store_true')
    parser.add_argument('-beid', "--batch_eval_id", type=int, default=0)
    parser.add_argument('-wd', "--weight_decay", type=float, default=10.0)

    # FedMD
    parser.add_argument('-mdls', "--fedmd_epoch", type=int, default=1)

    # FedDF
    parser.add_argument('-wdf', "--add_model_fusion", action='store_true')

    parser.add_argument('-dbg', "--debug", action='store_true', help='for debug')
    parser.add_argument('-vis', "--visualization", action='store_true', help='for visualization')
    parser.add_argument('-v_loss', "--vis_cw_loss", action='store_true')

    # FedCST
    parser.add_argument('-wrl', "--random_label", action='store_true')

    parser.add_argument('-ps', "--public_size", type=int, default=5000)
    parser.add_argument('-pdt', "--public_dataset", type=str, default='Cifar100')
    parser.add_argument('-wcst', "--add_class_specific_teacher", action='store_true')
    parser.add_argument('-scst', "--selected_class_specific_teacher", action='store_true')
    parser.add_argument('-ncst', "--num_class_specific_teacher", type=float, default=0.001)
    parser.add_argument('-tcst', "--cst_time", type=int, default=0)
    parser.add_argument('-lrcst', "--cst_learning_rate", type=float, default=1e-3)
    parser.add_argument('-bscst', "--cst_batch_size", type=int, default=128)
    parser.add_argument('-losscst', "--cst_loss", type=str, default='mse')
    parser.add_argument('-optcst', "--cst_optimizer", type=str, default='adam')
    parser.add_argument('-lscst', "--cst_epoch", type=int, default=1)
    parser.add_argument('-wls', "--add_logit_shaper", type=str, default='no')
    parser.add_argument('-lswd', "--ls_weight_decay", type=float, default=0.001)

    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")

    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.1)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1e3,
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1e-1)
    # APFL
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON
    parser.add_argument('-tau', "--tau", type=float, default=1.0)

    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)

    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.01)
    parser.add_argument('-L', "--L", type=float, default=0.2)

    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)

    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)

    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)

    # FedAvgDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        if platform.system() == 'Linux':
            if not torch.cuda.is_available():
                args.device = "cpu"
        else:
            if torch.backends.mps.is_available():
                args.device = "mps"

    if args.device == 'cuda':
        args.device = args.device + ":" + args.device_id

    DATA_PATH = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    file_path = os.path.join(dataset_path, 'config.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
        args.num_classes = data['num_classes']
        args.dataset_type = data['partition']
        data_statistic = data['Size of samples for labels in clients']

    data_dist = np.zeros((args.num_clients, args.num_classes))
    for i in range(args.num_clients):
        client_data_dist_np = np.array(data_statistic[i])
        data_dist[i, client_data_dist_np[:, 0]] = client_data_dist_np[:, 1]

    algo = args.dataset + "_" + args.algorithm
    result_path = "./results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"Directory '{result_path}' created.")
    algo = algo + "_" + args.goal
    file_path = result_path + "{}_data_distribution.npy".format(algo)
    np.save(file_path, data_dist)
    args.data_dist = data_dist.tolist()

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Local learning rate: {}".format(args.local_learning_rate))
    print("Local learning rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learning rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Partition: {}".format(args.dataset_type))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model))
    print("Using device: {}".format(args.device))
    print("Using DP: {}".format(args.privacy))
    if args.privacy:
        print("Sigma for DP: {}".format(args.dp_sigma))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))

    if args.algorithm == 'cwFedAvg':
        print("Class-wise aggregation: {}".format(args.add_cw))
        print("True class distribution based aggregation: {}".format(args.use_true_dist))
        print("Use weight regularizer: {}".format(args.add_wdr))
        print("Weigh regularizer decay: {}".format(args.weight_decay))

    if args.algorithm == 'FedCST':
        print("Class Specific Teacher: {}".format(args.add_class_specific_teacher))
        if args.selected_class_specific_teacher:
            print("Number of Class Specific Teacher: {}".format(args.num_class_specific_teacher))
        else:
            print("Number of Class Specific Teacher: ALL")
        print("Random Public Data Label: {}".format(args.random_label))
        print("CST Loss: {}".format(args.cst_loss))
        print("CST Optimizer: {}".format(args.cst_optimizer))
        print("CST Time: {}".format(args.cst_time))
        print("CST batch size: {}".format(args.cst_batch_size))
        print("CST learning rate: {}".format(args.cst_learning_rate))
        print("CST training epoch: {}".format(args.cst_epoch))
        print("Number of Public Data: {}".format(args.public_size))
        print("Logit shaper: {}".format(args.add_logit_shaper))
        print("Logit shaper weight decay: {}".format(args.ls_weight_decay))

    print("=" * 50)

    args_dict = vars(args)
    file_name = args.goal + '_args.json'
    with open(file_name, "w") as f:
        json.dump(args_dict, f, indent=4)

    run(args)

