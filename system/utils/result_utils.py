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

import h5py
import numpy as np
import os
import matplotlib.pyplot as plt



def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "./results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


def plotting_trial_result(conf):
    file_path = conf.goal + "_result.h5"
    result_data = {}
    with h5py.File(file_path, 'r') as hf:
        result_data['acc'] = np.array(hf.get('rs_test_acc'))
        result_data['loss'] = np.array(hf.get('rs_train_loss'))

    for key, value in result_data.items():
        plt.figure(figsize=(10, 5))
        plt.plot(value, marker='o', linestyle='-', color='b', label=key)
        plt.title(conf.goal + '_' + key)
        plt.xlabel('Communication round')
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        result_name = conf.goal + '_' + key
        plt.savefig(result_name + '.png')

    if conf.vis_cw_loss:
        with h5py.File(file_path, 'r') as hf:
            cw_train_loss = np.array(hf.get('rs_cw_train_loss'))

        for id in range(conf.num_clients):
            value = cw_train_loss[:,id]
            plt.figure(figsize=(10, 5))
            plt.plot(value, marker='o', linestyle='-', color='b', label=key)
            plt.title(conf.goal + f'_train_loss_id_{id}')
            plt.xlabel('Communication round')
            plt.ylabel(key)
            plt.legend()
            plt.grid(True)
            result_name = conf.goal + f'_train_loss_id_{id}'
            plt.savefig(result_name + '.png')

    # file_name = conf.dataset + "_" + conf.algorithm + "_" + conf.goal + "_0"
    # file_path = "./results/" + file_name + ".h5"
    # result_data = {}
    # with h5py.File(file_path, 'r') as hf:
    #     result_data['acc'] = np.array(hf.get('rs_test_acc'))
    #     result_data['loss'] = np.array(hf.get('rs_train_loss'))
    #
    # for key, value in result_data.items():
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(value, marker='o', linestyle='-', color='b', label=key)
    #     plt.title(file_name + '_' + key)
    #     plt.xlabel('Communication round')
    #     plt.ylabel(key)
    #     plt.legend()
    #     plt.grid(True)
    #     result_name = file_name + '_' + key
    #     plt.savefig(result_name + '.png')
