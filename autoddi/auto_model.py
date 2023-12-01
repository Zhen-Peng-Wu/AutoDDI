import os
from autoddi.estimation import Scratch_Train_Test
from autoddi.search_space.search_space_config import SearchSpace
import torch
from autoddi.search_algorithm.graphnas_search_algorithm import Search

class AutoModel(object):
    """
    The top API to realize gnn architecture search and model testing automatically.

    Using search algorithm samples gnn architectures and evaluate
    corresponding performance,testing the top k model from the sampled
    gnn architectures based on performance.

    Args:
        data: graph data obj
            the target graph data object including required attributes:

        search_parameter: dict
            the search algorithm configuration dict to control the
            automatic search process including required attributes:
            1.search_algorithm_type, 2.test_gnn_num

        gnn_parameter: dict
            the gnn configuration dict to complete the gnn model train
            validate and test based on the gnn architecture, for the
            stack gcn architecture, the required attributes includes:

    Returns:
        None
    """

    def __init__(self, data, search_parameter, gnn_parameter, save_suffix):

        self.data = data
        self.search_parameter = search_parameter
        self.gnn_parameter = gnn_parameter
        self.search_space = SearchSpace()
        self.save_suffix = save_suffix

        print("search parameter information:\t", self.search_parameter)
        print("gnn parameter information:\t", self.gnn_parameter)
        print("stack gcn architecture information:\t", self.search_space.stack_gcn_architecture)

        self.search_algorithm = Search(self.data,
                                       self.search_parameter,
                                       self.gnn_parameter,
                                       self.search_space)

        if self.gnn_parameter['mode'] == 'search':
            self.search_model()
        ## if mode == test, not search, but you should ensure the logger dir has exist in the disk

        self.derive_target_model()

    def search_model(self):

        self.search_algorithm.search_operator()

    def derive_target_model(self):

        path = os.path.split(os.path.realpath(__file__))[0][:-(len('autoddi')+1)] + "/logger"+'/' + str(self.data.data_logger_save)+"/gnn_logger_"
        architecture_performance_list = self.gnn_architecture_performance_load(path, self.data.data_logger_save)
        gnn_architecture_performance_dict = {}
        gnn_architecture_list = []
        performance_list = []

        for line in architecture_performance_list:
            line = line.split(":")
            gnn_architecture = eval(line[0])
            performance = eval(line[1].replace("\n", ""))
            gnn_architecture_list.append(gnn_architecture)
            performance_list.append(performance)

        for key, value in zip(gnn_architecture_list, performance_list):
            gnn_architecture_performance_dict[str(key)] = value

        ranked_gnn_architecture_performance_dict = sorted(gnn_architecture_performance_dict.items(),
                                                          key=lambda x: x[1],
                                                          reverse=True)

        sorted_gnn_architecture_list = []
        sorted_performance = []

        top_k = int(self.search_parameter["test_gnn_num"])
        i = 0
        for key, value in ranked_gnn_architecture_performance_dict:
            if i == top_k:
                break
            else:
                sorted_gnn_architecture_list.append(eval(key))
                sorted_performance.append(value)
                i += 1

        print(35*"=" + " the testing start " + 35*"=")

        from planetoid import Planetoid
        folds = ['fold0', 'fold1', 'fold2']
        model_num = [num for num in range(len(sorted_gnn_architecture_list))]

        for target_architecture, num in zip(sorted_gnn_architecture_list, model_num):
            print("test gnn architecture:\t", str(target_architecture))
            performance_all = []
            for fold in folds:
                self.data = Planetoid(self.data.data_name, fold=fold, save_suffix=self.save_suffix)
                performance = Scratch_Train_Test(target_architecture, num, fold, self.data, self.gnn_parameter, self.search_parameter['device'])
                performance_all.append(performance)

            if self.data.transductive_flag:
                ### transductive
                test_acc = torch.tensor([p[0]*100 for p in performance_all])
                test_auc_roc = torch.tensor([p[1]*100 for p in performance_all])
                test_f1 = torch.tensor([p[2]*100 for p in performance_all])
                test_ap = torch.tensor([p[-1]*100 for p in performance_all])

                print("all fold test_acc:\t"+str(round(test_acc.mean().item(),2))+"±" +str(round(test_acc.std().item(),2)))
                print("all fold test_auc_roc:\t" + str(round(test_auc_roc.mean().item(),2)) + "±" + str(round(test_auc_roc.std().item(),2)))
                print("all fold test_f1:\t" + str(round(test_f1.mean().item(),2)) + "±" + str(round(test_f1.std().item(),2)))
                print("all fold test_ap:\t" + str(round(test_ap.mean().item(),2)) + "±" + str(round(test_ap.std().item(),2)))
            else:
                ### inductive
                s1_acc = torch.tensor([p[0][0] * 100 for p in performance_all])
                s1_auc_roc = torch.tensor([p[0][1] * 100 for p in performance_all])
                s1_f1 = torch.tensor([p[0][2] * 100 for p in performance_all])
                s1_ap = torch.tensor([p[0][-1] * 100 for p in performance_all])

                print("all fold s1_acc:\t" + str(round(s1_acc.mean().item(), 2)) + "±" + str(round(s1_acc.std().item(), 2)))
                print("all fold s1_auc_roc:\t" + str(round(s1_auc_roc.mean().item(), 2)) + "±" + str(round(s1_auc_roc.std().item(), 2)))
                print("all fold s1_f1:\t" + str(round(s1_f1.mean().item(), 2)) + "±" + str(round(s1_f1.std().item(), 2)))
                print("all fold s1_ap:\t" + str(round(s1_ap.mean().item(), 2)) + "±" + str(round(s1_ap.std().item(), 2)))

                s2_acc = torch.tensor([p[1][0] * 100 for p in performance_all])
                s2_auc_roc = torch.tensor([p[1][1] * 100 for p in performance_all])
                s2_f1 = torch.tensor([p[1][2] * 100 for p in performance_all])
                s2_ap = torch.tensor([p[1][-1] * 100 for p in performance_all])

                print("all fold s2_acc:\t" + str(round(s2_acc.mean().item(), 2)) + "±" + str(round(s2_acc.std().item(), 2)))
                print("all fold s2_auc_roc:\t" + str(round(s2_auc_roc.mean().item(), 2)) + "±" + str(round(s2_auc_roc.std().item(), 2)))
                print("all fold s2_f1:\t" + str(round(s2_f1.mean().item(), 2)) + "±" + str(round(s2_f1.std().item(), 2)))
                print("all fold s2_ap:\t" + str(round(s2_ap.mean().item(), 2)) + "±" + str(round(s2_ap.std().item(), 2)))

        print(35 * "=" + " the testing ending " + 35 * "=")

    def gnn_architecture_performance_load(self, path, data_logger_save):

        with open(path + data_logger_save + ".txt", "r") as f:
            gnn_architecture_performance_list = f.readlines()
        return gnn_architecture_performance_list

