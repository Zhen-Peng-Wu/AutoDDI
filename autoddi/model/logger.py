import os
import torch
import time

logger_path = os.path.split(os.path.realpath(__file__))[0][:-(7+len('autoddi'))] + "/logger"

def gnn_architecture_performance_save(gnn_architecture, performance, data_logger_save):

    if not os.path.exists(logger_path+ '/' + str(data_logger_save)):
        os.makedirs(logger_path+ '/' + str(data_logger_save))

    with open(logger_path + '/' + str(data_logger_save) + "/gnn_logger_" + str(data_logger_save) + ".txt", "a+") as f:
        f.write(str(gnn_architecture) + ":" + str(performance) + "\n")

    print("gnn architecture and validation performance save")
    print("save path: ", logger_path + '/' + str(data_logger_save)+ "/gnn_logger_" + str(data_logger_save) + ".txt")
    print(50 * "=")

def test_performance_save(gnn_architecture, test_performance_dict, hyperparameter_dict, data_logger_save):

    if not os.path.exists(logger_path+ '/' + str(data_logger_save)):
        os.makedirs(logger_path+ '/' + str(data_logger_save))

    file_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

    with open(logger_path+ '/' + str(data_logger_save) + "/test_logger_" + data_logger_save + "_" + file_time + ".txt", "a+") as f:
        f.write("gnn architecture:\t" + str(gnn_architecture)+ "\n")
        f.write(25 * "=" + " hyperparameter " + 25 * "=" + "\n")
        for hyperparameter in hyperparameter_dict.keys():
            f.write(str(hyperparameter) + ":" + str(hyperparameter_dict[hyperparameter])+"\n")
        f.write(25*"=" + " test performance result " + 25*"=" + "\n")
        for performance in test_performance_dict.keys():
            f.write(str(performance) + ":" + str(test_performance_dict[performance])+"\n")
        f.write(50 * "=" + "\n\n")

    print("hyperparameter and test performance save")
    print("save path: ", logger_path+ '/' + str(data_logger_save) + "/test_logger_" + data_logger_save + "_" + file_time + ".txt")

def model_save(model, optimizer, data_logger_save, model_num, fold):
    torch.save(model, logger_path+ '/' + str(data_logger_save)+"/model_" + data_logger_save + "_num" + str(model_num)+'_'+str(fold) + ".pkl")

def model_load(data_logger_save, model_num, fold):
    model = torch.load(logger_path + '/' + str(data_logger_save) + "/model_" + data_logger_save + "_num" + str(model_num) + '_' + str(fold) + ".pkl")
    return model
