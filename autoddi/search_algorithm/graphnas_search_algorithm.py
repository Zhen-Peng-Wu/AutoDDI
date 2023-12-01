import torch
import torch.nn.functional as F
import numpy as np
import time
import scipy.signal
import os
from autoddi.estimation import Estimation
from autoddi.search_algorithm import utils

class SearchController(torch.nn.Module):

    def __init__(self,
                 search_parameter,
                 search_space,
                 action_list,
                 controller_hid=100,
                 cuda=True,
                 mode="train",
                 softmax_temperature=5.0,
                 tanh_c=2.5):

        super(SearchController, self).__init__()
        self.mode = mode
        # search_algorithm space or operators set containing operators used to build GNN
        self.search_space = search_space
        # operator categories for each controller RNN output
        self.action_list = action_list
        self.controller_hid = controller_hid
        self.is_cuda = cuda

        # set hyperparameters
        if search_parameter and search_parameter["softmax_temperature"]:
            self.softmax_temperature = float(search_parameter["softmax_temperature"])
        else:
            self.softmax_temperature = softmax_temperature

        if search_parameter and search_parameter["tanh_c"]:
            self.tanh_c = float(search_parameter["tanh_c"])
        else:
            self.tanh_c = tanh_c

        # build encoder
        self.num_tokens = []
        for key in self.search_space:
            self.num_tokens.append(len(self.search_space[key]))

        num_total_tokens = sum(self.num_tokens)  # count action type
        self.encoder = torch.nn.Embedding(num_total_tokens, controller_hid)

        # the core of controller
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        # build decoder
        self._decoders = torch.nn.ModuleDict()
        for key in self.search_space:
            size = len(self.search_space[key])
            decoder = torch.nn.Linear(controller_hid, size)
            self._decoders[key] = decoder

        self.reset_parameters()

    def _construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            for action, action_name in zip(single_action, self.action_list):
                predicted_actions = self.search_space[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self._decoders:
            self._decoders[decoder].bias.data.fill_(0)

    def forward(self,
                inputs,
                hidden,
                action_name,
                is_embed):

        embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self._decoders[action_name](hx)

        logits /= self.softmax_temperature

        # exploration
        if self.mode == 'train':
            logits = (self.tanh_c * torch.tanh(logits))

        return logits, (hx, cx)

    def action_index(self, action_name):
        key_names = self.search_space.keys()
        for i, key in enumerate(key_names):
            if action_name == key:
                return i

    def sample(self, batch_size=1, device='cuda:0'):

        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')

        inputs = torch.zeros([batch_size, self.controller_hid])
        hidden = (torch.zeros([batch_size, self.controller_hid]), torch.zeros([batch_size, self.controller_hid]))
        # if self.is_cuda:
        #     inputs = inputs.cuda()
        #     hidden = (hidden[0].cuda(), hidden[1].cuda())
        if self.is_cuda:
            inputs = inputs.to(device)
            hidden = (hidden[0].to(device), hidden[1].to(device))

        entropies = []
        log_probs = []
        actions = []

        for block_idx, action_name in enumerate(self.action_list):
            decoder_index = self.action_index(action_name)

            logits, hidden = self.forward(inputs,
                                          hidden,
                                          action_name,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, device=device, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:decoder_index]),
                self.is_cuda,
                device=device,
                requires_grad=False)

            inputs = self.encoder(inputs)
            actions.append(action[:, 0])

        actions = torch.stack(actions).transpose(0, 1)
        dags = self._construct_action(actions)

        return dags, torch.cat(log_probs), torch.cat(entropies)

class MoveAverageOperator(object):

    def __init__(self, top_k=10):
        self.scores = []
        self.top_k = top_k

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0

        self.scores.append(score)
        self.scores.sort(reverse=True)
        self.scores = self.scores[:self.top_k]
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)

class Search(object):

    def __init__(self,
                 data,
                 search_parameter,
                 gnn_parameter,
                 search_space):

        self.cuda = search_parameter["cuda"]
        self.epoch = 0
        self.start_epoch = 0
        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter
        self.history = []
        self.baseline = None
        self.search_space = search_space
        self.device = self.search_parameter['device']

        # build move average reward operator
        self.move_average_reward_operator = MoveAverageOperator()
        # build LSTM SearchController
        self.controller = SearchController(self.search_parameter,
                                           action_list=self.search_space.stack_gcn_architecture,
                                           search_space=self.search_space.space_dict,
                                           cuda=self.cuda).to(self.device)
        # if self.cuda:
        #     self.controller.cuda()
        ### build LSTM optimizer
        self.controller_optim = torch.optim.Adam(self.controller.parameters(),
                                                 lr=float(self.search_parameter["controller_lr"]))

    def search_operator(self):

        start_time = time.time()
        print("*" * 35, "controller training start", "*" * 35)

        for self.epoch in range(self.start_epoch, int(self.search_parameter["controller_train_epoch"])):

            print("*" * 35, "the " + str(self.epoch+1) + "th training epoch for controller", "*" * 35, "\n")
            self.train_controller()

        print("*" * 35, "controller training ending", "*" * 35, "\n")

        print("*" * 35, "use controller sample model start", "*" * 35,"\n")
        self.search_based_on_trained_LSTM(int(self.search_parameter["search_scale"]))
        print("*" * 35, "use controller sampling ending", "*" * 35)
        search_total_time = time.time() - start_time

        path = os.path.split(os.path.realpath(__file__))[0][:-(17+len('autoddi'))]+'logger'+'/'+str(self.data.data_logger_save)+'/graphnas_logger'
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save(path,
                                   self.data.data_logger_save + "_search_total_time.txt",
                                   int(self.search_parameter["controller_train_epoch"]) +
                                   int(self.search_parameter["controller_train_parallel_num"]),
                                   int(self.search_parameter["search_scale"]),
                                   search_total_time)

    def scale(self, value, last_k=10, scale_value=1):
        '''
        scale value into [-scale_value, scale_value], according last_k history
        '''
        # find the large number in multiple lists
        max_reward = np.max(self.history[-last_k:])
        if max_reward == 0:
            return value
        return scale_value / max_reward * value

    def discount(self, x, amount):
        return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]

    def get_reward(self, gnn_architecture_list, entropies_list):
        """
        compute multiple rewards as list based on multiple sampled models on validation data.
        """

        ### GNNs estimation
        result = []
        for gnn_architecture in gnn_architecture_list:
            performance = Estimation(gnn_architecture=gnn_architecture,
                                     graph_data=self.data,
                                     gnn_parameter=self.gnn_parameter,
                                     device=self.search_parameter['device']
                                     )
            result.append(performance)

        # get multiple reward based on move average strategy
        move_average_reward_list = []
        for val_acc in result:
            move_average_reward_list.append(self.move_average_reward_operator.get_reward(val_acc))

        rewards_list = []
        for index in range(len(move_average_reward_list)):
            reward_list = []
            reward_list.append(move_average_reward_list[index])
            rewards_list.append(reward_list + float(self.search_parameter["entropy_coeff"]) * entropies_list[index])

        return rewards_list

    def multiple_reward_mean(self, rewards_list):
        """
        calculate the mean reward list based on dimension i of rewards_list
        """
        mean_rewards = []
        for index_j in range(len(rewards_list[0])):
            temp_list = []
            for index_i in range(len(rewards_list)):
                temp_list.append(rewards_list[index_i][index_j])
            mean_rewards.append(np.mean(temp_list))
        mean_rewards = np.array(mean_rewards)
        return mean_rewards

    def select_max_reward_vector_index(self, rewards_list):

        for index in range(len(rewards_list)):
            if index == 0:
                max_reward_vector = np.sum(rewards_list[index])
                max_index = index
            elif max_reward_vector < np.sum(rewards_list[index]):
                max_reward_vector = np.sum(rewards_list[index])
                max_index = index
        return max_index

    def train_controller(self):
        """
        train controller to find better GNN architecture.
        """
        model = self.controller
        model.train()
        gnn_architecture_list = []
        log_probs_list = []
        entropies_list = []

        for step in range(int(self.search_parameter["controller_train_parallel_num"])):

            # controller sample a GNNs and get training information of LSTM
            gnn_architecture, log_probs, entropies = self.controller.sample(device=self.device)
            np_entropies = entropies.data.cpu().numpy()
            gnn_architecture_list.append(gnn_architecture[0])
            log_probs_list.append(log_probs)
            entropies_list.append(np_entropies)

        print("*" * 35, "controller sample model architectures:", "*" * 35, "\n")
        for gnn_architecture in gnn_architecture_list:
            print(gnn_architecture)

        # get multiple reward list as rewards_list based on multiple GNN architectures
        rewards_list = self.get_reward(gnn_architecture_list,
                                       entropies_list)
        torch.cuda.empty_cache()

        # get discount rewards_list
        if 1 > float(self.search_parameter["discount"]) > 0:
            discount_rewards_list = []
            for index in range(len(rewards_list)):
                discount_rewards_list.append(self.discount(rewards_list[index],
                                                           float(self.search_parameter["discount"])))
            rewards_list = discount_rewards_list

        # get moving average baseline
        # select max reward vector index from rewards_list
        max_index = self.select_max_reward_vector_index(rewards_list)
        if self.baseline is None:
            self.baseline = rewards_list[max_index]
        else:
            decay = float(self.search_parameter["ema_baseline_decay"])
            self.baseline = decay * self.baseline + (1 - decay) * rewards_list[max_index]

        adv_list = []
        for rewards in rewards_list:
            adv_list.append(rewards - self.baseline)

        self.history.append(rewards_list[max_index])

        # accumulation a adv_list gradient for optimizer
        for index in range(len(adv_list)):
            adv = self.scale(adv_list[index], scale_value=0.5)
            adv = utils.get_variable(adv, self.cuda, device=self.device, requires_grad=False)
            # calculate every reward policy loss
            loss = -log_probs_list[index] * adv
            loss = loss.sum()
            # calculate gradient based on loss backward
            loss.backward()

        # clip gradient for LSTM model parameter to prevent gradient disappearing
        if float(self.search_parameter["controller_grad_clip"]) > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(),
                                          float(self.search_parameter["controller_grad_clip"]))
        # update LSTM parameter based on optimizer
        self.controller_optim.step()
        self.controller_optim.zero_grad()
        torch.cuda.empty_cache()


    def search_based_on_trained_LSTM(self, search_scale):
        """
        sample model architecture based on trained LSTM
        """
        gnn_architecture_list, _, _ = self.controller.sample(search_scale, device=self.device)

        print("*" * 35, "sample model architecture list:", "*" * 35, "\n")
        for gnn_architecture in gnn_architecture_list:
            print(gnn_architecture)

        ### GNNs estimation
        result = []
        for gnn_architecture in gnn_architecture_list:
            performance = Estimation(gnn_architecture=gnn_architecture,
                                     graph_data=self.data,
                                     gnn_parameter=self.gnn_parameter,
                                     device=self.search_parameter['device']
                                     )
            result.append(performance)


        path = os.path.split(os.path.realpath(__file__))[0][:-(17+len('autoddi'))]+'logger'+'/'+str(self.data.data_logger_save)+'/graphnas_logger'
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_data_save(path, self.data.data_logger_save + ".txt", gnn_architecture_list, result)

        index = result.index(max(result))
        best_val_architecture = gnn_architecture_list[index]
        best_acc = max(result)
        print("\n", "*" * 35, "The Result", "*" * 35, "\n")
        print("Best architecture:", best_val_architecture)
        print("Best val_acc:", best_acc, "\n")




