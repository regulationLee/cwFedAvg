import torch
import os
import numpy as np
import csv
import h5py
import copy
import time
import random
from torch.utils.data import DataLoader
from utils.data_utils import read_client_data, read_public_data
from utils.dlg import DLG


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.round_cnt = 0
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate

        if isinstance(args.model, list):
            pass
        else:
            self.global_model = copy.deepcopy(args.model)

        self.fine_tune = False

        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.batches_scores = []
        self.batches_indices = []
        self.client_fc_norm = {}
        self.consensus = []
        self.consensus_indices = []
        self.subset_idx = []

        self.uploaded_weights = []
        self.uploaded_weights_cw = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_cw_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = \
            np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def receive_consensus(self):
        def load_public_data(subset_idx, batch_size=None):
            if batch_size == None:
                batch_size = self.batch_size
                batch_size = 128
            public_data = read_public_data(self.dataset, subset_idx, is_train=True)
            return DataLoader(public_data, batch_size, drop_last=True, shuffle=False)

        self.batches_scores = []
        self.batches_indices = []
        self.subset_idx = random.sample(range(50000), 5000)  # 55,000: MNIST, 50,000: Cifar 10/100
        self.public_loader = load_public_data(self.subset_idx)
        for c in self.selected_clients:
            tmp_consensus, tmp_consensus_indices = c.calc_consensus(self.public_loader)
            self.batches_scores.append(tmp_consensus)
            self.batches_indices.append(tmp_consensus_indices)

    def aggregate_consensus(self):
        self.consensus = []
        for scores in zip(*self.batches_scores):
            self.consensus.append(torch.stack(scores, dim=-1).mean(dim=-1).cpu())
        print('debug')

    def send_consensus(self):
        for c in self.selected_clients:
            c.set_consensus(self.consensus)

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
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
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)

    # # original code
    # def save_results(self):
    #     algo = self.dataset + "_" + self.algorithm
    #     result_path = "./results/"
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #
    #     if (len(self.rs_test_acc)):
    #         algo = algo + "_" + self.goal + "_" + str(self.times)
    #         file_path = result_path + "{}.h5".format(algo)
    #         print("File path: " + file_path)
    #
    #         with h5py.File(file_path, 'w') as hf:
    #             hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
    #             hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
    #             hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    # added by G.Lee
    def save_results(self):
        if (len(self.rs_test_acc)):
            file_path = self.goal + "_result.h5"
            print("\nFile path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                if self.args.vis_cw_loss:
                    hf.create_dataset('rs_cw_train_loss', data=self.rs_cw_train_loss)

            data_to_add = {
                "exp_name": self.args.goal,
                "accuracy": max(self.rs_test_acc),
                "avg_time": sum(self.Budget[1:]) / len(self.Budget[1:]),
            }
            print(f"\nBest accuracy: {data_to_add['accuracy']:.4f}")
            print(f"Average Process time: {data_to_add['avg_time']:.4f}")

            result_file = "[results].csv"
            with open(result_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(data_to_add.values())
                print(f"\nEXP: {self.args.goal} recorded.")



    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        self.round_cnt += 1

        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])

        cw_train_loss = [a / b if b != 0 else None for a, b in zip(stats_train[2], stats_train[1])]
        self.rs_cw_train_loss.append(cw_train_loss)

        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        # print('chk')
        # print(f'round_cnt: {self.round_cnt}')
        # print(f'global_rounds: {self.global_rounds}')
        # if self.round_cnt == self.global_rounds + 1:
        #     self.plotting_client_layer()
        #     if self.args.add_cw:
        #         self.plotting_global_layer()

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

            # items.append((client_model, origin_grad, target_inputs))

        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=False,
                               send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    def plotting_client_layer(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "./results/"
        algo = algo + "_" + self.goal
        if self.fine_tune:
            file_path = result_path + "{}_local_models_weight_fine_tune.npy".format(algo)
        else:
            file_path = result_path + "{}_local_models_weight.npy".format(algo)

        if self.args.add_cw:
            if self.args.add_wdr:
                file_path = result_path + "{}_wd_{}_local_models_weight.npy".format(algo, self.args.weight_decay)
            else:
                if self.args.use_true_dist:
                    file_path = result_path + "{}_gt_wo_wdr_local_models_weight.npy".format(algo)
                else:
                    file_path = result_path + "{}_wo_wdr_local_models_weight.npy".format(algo)

        concat_weight = torch.zeros([self.args.num_clients, self.args.num_classes])

        for client in self.selected_clients:
            if hasattr(client.model, 'head'):
                fc_weight_norm = torch.norm(client.model.head.weight, dim=1).unsqueeze(0)
            else:
                fc_weight_norm = torch.norm(client.model.fc.weight, dim=1).unsqueeze(0)
            concat_weight[client.id, :] = fc_weight_norm

        numpy_client_weight = concat_weight.detach().cpu().numpy()
        print('save client weight')
        print(file_path)
        np.save(file_path, numpy_client_weight)

    def plotting_global_layer(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = "./results/"
        algo = algo + "_" + self.goal
        if self.args.add_wdr:
            file_path = result_path + "{}_wd_{}_global_models_weight.npy".format(algo, self.args.weight_decay)
        else:
            if self.args.use_true_dist:
                file_path = result_path + "{}_gt_wo_wdr_global_models_weight.npy".format(algo)
            else:
                file_path = result_path + "{}_wo_wdr_global_models_weight.npy".format(algo)

        concat_weight = torch.zeros([self.args.num_classes, self.args.num_classes])

        for i in range(len(self.cw_global_model)):
            global_model = self.cw_global_model[i]
            if hasattr(global_model, 'head'):
                fc_weight_norm = torch.norm(global_model.head.weight, dim=1).unsqueeze(0)
            else:
                fc_weight_norm = torch.norm(global_model.fc.weight, dim=1).unsqueeze(0)

            concat_weight[i, :] = fc_weight_norm

        numpy_global_weight = concat_weight.detach().cpu().numpy()
        np.save(file_path, numpy_global_weight)

