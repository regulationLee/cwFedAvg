import time
import random
import copy
import numpy as np
import torch
from flcore.clients.clientavgCW import cwclientAVG
from flcore.servers.serverbase import Server
from collections import defaultdict


class cwFedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        if isinstance(args.model, list):
            pass
        else:
            if self.args.add_cw:
                self.cw_global_model = []
                for i in range(self.num_classes):
                    if self.args.partial_layer_train:
                        self.layer_groups = args.layer_groups
                        cw_layers = {}
                        for name, module in args.model.named_children():
                            if name in self.layer_groups["cw"]:
                                cw_layers[name] = copy.deepcopy(module)
                        self.cw_global_model.append(cw_layers)
                    elif self.args.decision_layer_only:
                        self.cw_global_model.append(copy.deepcopy(args.model.head))
                    else:
                        self.cw_global_model.append(copy.deepcopy(args.model))


        # select slow clients
        self.set_slow_clients()
        self.set_clients(cwclientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        if not self.args.decision_layer_only and not self.args.partial_layer_train:
            self.global_model = None

        if self.args.add_proto:
            self.global_protos = [None for _ in range(args.num_classes)]

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_models()
            assert (len(self.clients) > 0)

            for client in self.clients:
                start_time = time.time()

                if self.args.add_cw:
                    client.local_initializtion_cw(self.cw_global_model, self.global_model)
                else:
                    client.set_parameters(self.global_model)

                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                if i == 0:
                    self.args.batch_eval = True
                else:
                    self.args.batch_eval = False
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            if self.args.add_proto:
                self.receive_protos()
                self.global_protos = proto_aggregation(self.uploaded_protos)
                self.send_protos()

            if self.args.add_cw:
                self.receive_models_cw()
            else:
                self.receive_models()

            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)

            if self.args.add_cw:
                if self.args.decision_layer_only or self.args.partial_layer_train:
                    self.aggregate_parameters()
                    self.aggregate_parameters_cw()
                else:
                    self.aggregate_parameters_cw()
            else:
                self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        self.save_results()
        # self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(cwclientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

        print("Final value of regularizer(||p-p^||)")
        datadist_list = []
        wdr_list = []
        sum_wdr = []
        sum_zero_class = []
        for client in self.clients:
            # # original cwFedAvg code
            if self.args.partial_layer_train:
                fc_weight_norm = torch.norm(client.model.fc.weight, dim=1).unsqueeze(0)
            else:
                fc_weight_norm = torch.norm(client.model.head.weight, dim=1).unsqueeze(0)

            fc_weight_norm = fc_weight_norm.view(-1)
            if self.args.clip_weight:
                fc_weight_norm[client.mask] = 0
            fc_weight = fc_weight_norm / torch.sum(fc_weight_norm)
            wd_regularizer = torch.norm(fc_weight - client.gt, p=2)

            gt_np = client.gt.detach().to('cpu').numpy()
            fc_weight_np = fc_weight.detach().to('cpu').numpy()
            wdr_val = wd_regularizer.detach().to('cpu').numpy()

            datadist_list.append(gt_np)
            wdr_list.append(fc_weight_np)

            sum_wdr.append(wdr_val)
            sum_zero_class.append(np.sum(fc_weight_np[gt_np == 0]))
        print("Average value of regularizer: {}".format(sum(sum_wdr)/len(sum_wdr)))
        print("Average value of sum of zero class: {}".format(sum(sum_zero_class) / len(sum_zero_class)))

        # algo = self.args.dataset + "_" + self.args.algorithm
        # result_path = "../results/"
        # if self.args.clip_weight:
        #     algo = algo + "_" + self.args.goal + '_' + self.args.dataset_type + '_' + str(self.args.hetero_beta) + '_clip_weight'
        # else:
        #     algo = algo + "_" + self.args.goal + '_' + self.args.dataset_type + '_' + str(self.args.hetero_beta)
        # file_path = result_path + "{}_class_distribution.npy".format(algo)
        # np.save(file_path, np.array(datadist_list))
        # file_path = result_path + "{}_fc_weight_norm.npy".format(algo)
        # np.save(file_path, np.array(wdr_list))

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

    def send_models_cw(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.local_initialization_cw(self.our_global_model)
            # client.set_parameters_cw()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models_cw(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_weights_cw = []
        self.uploaded_models = []
        self.uploaded_models_head = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_models.append(client.model)

                if self.args.partial_layer_train:
                    cw_layers = {}
                    for name, module in client.model.named_children():
                        if name in self.layer_groups["cw"]:
                            cw_layers[name] = copy.deepcopy(module)
                    self.uploaded_models_head.append(cw_layers)

                elif self.args.decision_layer_only:
                    self.uploaded_models_head.append(client.model.head)

                self.uploaded_weights.append(client.train_samples)
                if self.args.use_true_dist:
                    weight_list = client.data_dist
                else:
                    weight_list = client.aggregate_weight_calc()

                weight_list = [x / sum(weight_list) for x in weight_list]
                self.uploaded_weights_cw.append(weight_list)

        # store client weight of FedAVG
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters_cw(self):
        assert (len(self.uploaded_models) > 0)

        # original cwFedAvg code
        for ig in range(len(self.cw_global_model)):
            if self.args.decision_layer_only:
                self.cw_global_model[ig] = copy.deepcopy(self.uploaded_models_head[0])
                for param in self.cw_global_model[ig].parameters():
                    param.data.zero_()
            elif self.args.partial_layer_train:
                for name, module in self.cw_global_model[ig].items():
                    for param in module.parameters():
                        param.data.zero_()
            else:
                for param in self.cw_global_model[ig].parameters():
                    param.data.zero_()

        fedavg_weight = np.array(self.uploaded_weights).reshape(self.num_clients, 1)
        fedavg_weight = np.repeat(fedavg_weight, self.num_classes, axis=1)
        uploaded_weight_np = np.array(self.uploaded_weights_cw)
        uploaded_weight_np = uploaded_weight_np * fedavg_weight

        marginal_weight_np = np.tile(np.sum(uploaded_weight_np, axis=0), (self.num_clients, 1))
        normalized_weight = np.divide(uploaded_weight_np, marginal_weight_np).tolist()

        for idx in range(len(normalized_weight)):
            w = normalized_weight[idx]
            if self.args.partial_layer_train:
                local_model = self.uploaded_models_head[idx]

                # # original cwFedAvg code
                # for idg in range(len(self.cw_global_model)):
                #     for target, source in zip(self.cw_global_model[idg].parameters(),
                #                               local_model.parameters()):
                #         target.data += source.data.clone() * w[idg]

                for idg in range(len(self.cw_global_model)):
                    for name, module in self.cw_global_model[idg].items():
                        for target, source in zip(module.parameters(), local_model[name].parameters()):
                            target.data += source.data.clone() * w[idg]

            elif self.args.decision_layer_only:
                local_model = self.uploaded_models_head[idx]

                for idg in range(len(self.cw_global_model)):
                    for target, source in zip(self.cw_global_model[idg].parameters(),
                                              local_model.parameters()):
                        target.data += source.data.clone() * w[idg]

            else:
                local_model = self.uploaded_models[idx]

                for idg in range(len(self.cw_global_model)):
                    for target, source in zip(self.cw_global_model[idg].parameters(),
                                              local_model.parameters()):
                        target.data += source.data.clone() * w[idg]


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label