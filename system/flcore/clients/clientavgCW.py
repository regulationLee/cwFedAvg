import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from collections import defaultdict

class cwclientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        if self.args.partial_layer_train:
            self.layer_groups = args.layer_groups

            self.aggregate_global = {}
            for name, module in args.model.named_children():
                if name in self.layer_groups["cw"]:
                    self.aggregate_global[name] = copy.deepcopy(module)
        else:
            self.aggregate_global = copy.deepcopy(args.model)


        self.weight_decay = self.args.weight_decay
        self.data_dist = args.data_dist[id]
        gt = [element / sum(self.data_dist) for element in self.data_dist]
        self.gt = torch.tensor(gt).to(self.device)
        self.mask = (self.gt == 0)

        if self.args.split_train:
            if self.args.partial_layer_train:
                if self.args.cw_layer_num != 0:
                    self.optimizer_common = torch.optim.SGD(
                        sum([list(getattr(self.model, name).parameters()) for name in self.layer_groups["common"]], []),
                        lr=self.learning_rate
                    )
                self.optimizer_cw = torch.optim.SGD(
                    sum([list(getattr(self.model, name).parameters()) for name in self.layer_groups["cw"]], []),
                    lr=self.args.head_lr
                )
            else:
                self.optimizer_common = torch.optim.SGD(self.model.base.parameters(), lr=self.learning_rate)
                self.optimizer_cw = torch.optim.SGD(self.model.head.parameters(), lr=self.args.head_lr)

        if self.args.add_proto:
            self.protos = None
            self.global_protos = None
            self.loss_mse = nn.MSELoss()

            self.lamda = 1

    def train(self):
        # self.model.to(self.device)
        self.model.train()

        # differential privacy
        if self.privacy:
            model_origin = copy.deepcopy(self.model)
            self.model, self.optimizer, trainloader, privacy_engine = \
                initialize_dp(self.model, self.optimizer, trainloader, self.dp_sigma)

        if self.args.batch_eval and self.args.batch_eval_id == self.id:
            algo = self.dataset + "_" + self.algorithm
            result_path = "../results/"
            file_path = result_path + "{}_client_{}_local_models_weight.npy".format(algo, self.id)

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)


        if self.args.split_train:
            # train head
            if self.args.partial_layer_train:
                for name, layer in self.model.named_children():
                    if name in self.layer_groups["cw"]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    else:
                        for param in layer.parameters():
                            param.requires_grad = False
            elif self.args.decision_layer_only:
                for param in self.model.base.parameters():
                    param.requires_grad = False
                for param in self.model.head.parameters():
                    param.requires_grad = True

            trainloader = self.load_train_data(batch_size=self.args.head_bs)

            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if self.args.batch_eval and self.args.batch_eval_id == self.id:
                        if hasattr(self.model, 'head'):
                            fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)
                        else:
                            fc_weight_norm = torch.norm(self.model.fc.weight, dim=1).unsqueeze(0)
                        if i == 0:
                            concat_weight = fc_weight_norm
                        else:
                            concat_weight = torch.cat((concat_weight, fc_weight_norm), dim=0)

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    output = self.model(x)
                    loss = self.loss(output, y)

                    if self.args.add_wdr:
                        if self.args.partial_layer_train:
                            fc_weight_norm = torch.norm(self.model.fc.weight, dim=1).unsqueeze(0)
                        else:
                            fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)

                        fc_weight_norm = fc_weight_norm.view(-1)
                        fc_weight = fc_weight_norm / torch.sum(fc_weight_norm)
                        wd_regularizer = torch.norm(self.gt - fc_weight, p=2)
                        loss += 0.5 * self.weight_decay * wd_regularizer

                    self.optimizer_cw.zero_grad()
                    loss.backward()
                    self.optimizer_cw.step()

            # train base
            if self.args.partial_layer_train:
                for name, layer in self.model.named_children():
                    if name in self.layer_groups["cw"]:
                        for param in layer.parameters():
                            param.requires_grad = False
                    else:
                        for param in layer.parameters():
                            param.requires_grad = True
            elif self.args.decision_layer_only:
                for param in self.model.base.parameters():
                    param.requires_grad = True
                for param in self.model.head.parameters():
                    param.requires_grad = False

            trainloader = self.load_train_data()
            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    if self.args.add_proto:
                        if self.args.model == "cnn":
                            out = self.conv1(x)
                            out = self.conv2(out)
                            out = torch.flatten(out, 1)
                            rep = self.last_feat(out)
                            output = self.fc(rep)
                        else:
                            raise NotImplementedError(f"Only {(self.args.model)} Model can be supported.")
                    else:
                        output = self.model(x)
                    loss = self.loss(output, y)

                    if self.args.add_proto:
                        if self.global_protos is not None:
                            proto_new = copy.deepcopy(rep.detach())
                            for i, yy in enumerate(y):
                                y_c = yy.item()
                                if type(self.global_protos[y_c]) != type([]):
                                    proto_new[i, :] = self.global_protos[y_c].data
                            loss += self.loss_mse(proto_new, rep) * self.lamda

                    self.optimizer_common.zero_grad()
                    loss.backward()
                    self.optimizer_common.step()
        else:
            trainloader = self.load_train_data()
            for epoch in range(max_local_epochs):
                for i, (x, y) in enumerate(trainloader):
                    if self.args.batch_eval and self.args.batch_eval_id == self.id:
                        if hasattr(self.model, 'head'):
                            fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)
                        else:
                            fc_weight_norm = torch.norm(self.model.fc.weight, dim=1).unsqueeze(0)
                        if i == 0:
                            concat_weight = fc_weight_norm
                        else:
                            concat_weight = torch.cat((concat_weight,fc_weight_norm), dim=0)

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)

                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))

                    output = self.model(x)
                    loss = self.loss(output, y)

                    if self.args.add_wdr:
                        if self.args.partial_layer_train:
                            fc_weight_norm = torch.norm(self.model.fc.weight, dim=1).unsqueeze(0)
                        else:
                            fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)

                        fc_weight_norm = fc_weight_norm.view(-1)
                        # if self.args.clip_weight:
                        #     fc_weight_norm[self.mask] = 0
                        fc_weight = fc_weight_norm / torch.sum(fc_weight_norm)
                        wd_regularizer = torch.norm(self.gt - fc_weight, p=2)
                        loss += 0.5 * self.weight_decay * wd_regularizer
                        # if self.args.add_logit_shaper:
                        #     logits_tensor = output.to(self.device)
                        #     _, _, logits_regularization = calculate_class_statistics(logits_tensor, y)
                        #     loss += 0.5 * self.args.ls_weight_decay * logits_regularization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        # if self.args.batch_eval and self.args.batch_eval_id == self.id:
        #     numpy_local_weight = concat_weight.detach().cpu().numpy()
        #     np.save(file_path, numpy_local_weight)

        if self.args.add_proto:
            self.collect_protos()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            eps, DELTA = get_dp_params(privacy_engine)
            print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")

            for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
                param.data = param_dp.data.clone()
            self.model = model_origin
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def aggregate_weight_calc(self):
        if self.args.partial_layer_train:
            fc_weight_norm = torch.norm(self.model.fc.weight, dim=1).unsqueeze(0)
        else:
            fc_weight_norm = torch.norm(self.model.head.weight, dim=1).unsqueeze(0)

        fc_weight_norm = fc_weight_norm.view(-1)
        # if self.args.clip_weight:
        #     fc_weight_norm[self.mask] = 0
        fc_weight_norm_list = fc_weight_norm.detach().cpu().numpy().tolist()
        return fc_weight_norm_list

    def local_initializtion_cw(self, received_cw_global_models, global_model=None):
        # calculate aggregate weight based on client's class distribution
        if self.args.use_true_dist:
            weight_list = self.data_dist
        else:
            weight_list = self.aggregate_weight_calc()
        weight_list = [element / sum(weight_list) for element in weight_list]
        # modified_list = [0 if element <= 0.0001 else element for element in weight_list]

        if self.args.partial_layer_train:
            for key, module in self.aggregate_global.items():
                for param in module.parameters():
                    param.data = torch.zeros_like(param.data)

            for w, cw_g_model in zip(weight_list, received_cw_global_models):
                for name, module in cw_g_model.items():
                    for agg_param, global_param in zip(self.aggregate_global[name].parameters(), module.parameters()):
                        agg_param.data += global_param.data.clone() * w
        else:
            self.aggregate_global = copy.deepcopy(received_cw_global_models[0])
            for param in self.aggregate_global.parameters():
                param.data = torch.zeros_like(param.data)

            for w, cw_g_model in zip(weight_list, received_cw_global_models):
                for agg_param, global_param in zip(self.aggregate_global.parameters(), cw_g_model.parameters()):
                    agg_param.data += global_param.data.clone() * w


        # local model aggregation
        if self.args.decision_layer_only:
            # # feature extractor update
            for new_param, old_param in zip(global_model.base.parameters(), self.model.base.parameters()):
                old_param.data = new_param.data.clone()
            # decision layer update
            for new_param, old_param in zip(self.aggregate_global.parameters(), self.model.head.parameters()):
                old_param.data = new_param.data.clone()
        elif self.args.partial_layer_train:
            for name, module in self.model.named_children():
                if name in self.layer_groups["cw"]:
                    for new_param, old_param in zip(self.aggregate_global[name].parameters(), module.parameters()):
                        old_param.data = new_param.data.clone()
                else:
                    for g_name, g_module in global_model.named_children():
                        if g_name in name:
                            for new_param, old_param in zip(g_module.parameters(), module.parameters()):
                                old_param.data = new_param.data.clone()
        else:
            for new_param, old_param in zip(self.aggregate_global.parameters(), self.model.parameters()):
                old_param.data = new_param.data.clone()

    # for add prototype learning
    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)


def calculate_class_statistics(output, target):
    # Get the number of classes
    num_classes = target.max().item() + 1

    # Convert target to one-hot encoding
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=num_classes).float()

    # Calculate class meansß
    class_means = torch.sum(output.unsqueeze(1) * target_one_hot.unsqueeze(2), dim=0) / (
                torch.sum(target_one_hot, dim=0).unsqueeze(1) + 1e-8)

    # Center the data
    centered_output = output.unsqueeze(1) - class_means.unsqueeze(0)

    # Calculate class-wise covariance matrices
    covariance_matrices = torch.sum(
        centered_output.unsqueeze(3) * centered_output.unsqueeze(2) * target_one_hot.unsqueeze(-1).unsqueeze(-1),
        dim=0) / (torch.sum(target_one_hot, dim=0).unsqueeze(-1).unsqueeze(-1) - 1 + 1e-8)

    # Calculate total variance using the trace of the covariance matrices
    total_variance = torch.sum(torch.diagonal(covariance_matrices, dim1=-2, dim2=-1))

    return class_means, covariance_matrices, total_variance

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos