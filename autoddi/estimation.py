import torch
from autoddi.model.gnn_model import GnnModel
from autoddi.model.logger import gnn_architecture_performance_save, model_save, model_load
import numpy as np
from autoddi.model.custom_loss import SigmoidLoss
from sklearn import metrics
from datetime import datetime
import time

def do_compute(batch, device, model):
    '''
        *batch: (pos_tri, neg_tri)
        *pos/neg_tri: (batch_h, batch_t, batch_r)
    '''
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch

    pos_tri = [tensor.to(device=device) for tensor in pos_tri]
    p_score = model(pos_tri)
    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score = model(neg_tri)
    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))

    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap


def eval_performance(eval_data_loader, eval_model, eval_device):
    eval_probas_pred = []
    eval_ground_truth = []
    with torch.no_grad():
        for batch in eval_data_loader:
            eval_model.eval()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, eval_device, eval_model)
            eval_probas_pred.append(probas_pred)
            eval_ground_truth.append(ground_truth)
        eval_probas_pred = np.concatenate(eval_probas_pred)
        eval_ground_truth = np.concatenate(eval_ground_truth)
        eval_acc, eval_auc_roc, eval_f1, eval_precision, eval_recall, eval_int_ap, eval_ap = do_compute_metrics(eval_probas_pred, eval_ground_truth)
    return eval_acc, eval_auc_roc, eval_f1, eval_precision, eval_recall, eval_int_ap, eval_ap


def Estimation(gnn_architecture,
               graph_data,
               gnn_parameter,
               device = "cuda:0"
               ):

    model = GnnModel(gnn_architecture,
                     graph_data.num_features,
                     graph_data.num_labels,
                     graph_data.rel_total,
                     graph_data.data_name).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=gnn_parameter['opt_type_dict']["learning_rate"],
                                 weight_decay=gnn_parameter['opt_type_dict']["l2_regularization_strength"])

    loss_fn = SigmoidLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    return_performance = 0
    for epoch in range(1, int(gnn_parameter['train_epoch']) + 1):
        for batch in graph_data.train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_probas_pred = []
        val_ground_truth = []
        with torch.no_grad():
            for batch in graph_data.val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)

            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, val_precision, val_recall, val_int_ap, val_ap = do_compute_metrics(val_probas_pred, val_ground_truth)
            if val_acc > return_performance:
                return_performance = val_acc
    gnn_architecture_performance_save(gnn_architecture, return_performance, graph_data.data_logger_save)
    return return_performance

def Scratch_Train_Test(gnn_architecture, num, fold, graph_data, gnn_parameter, device = "cuda:0"):

    model = GnnModel(gnn_architecture,
                     graph_data.num_features,
                     graph_data.num_labels,
                     graph_data.rel_total,
                     graph_data.data_name).to(device)

    optimizer = torch.optim.Adam([{'params': model.parameters()}],
                                 lr=gnn_parameter['opt_type_dict']["learning_rate"],
                                 weight_decay=gnn_parameter['opt_type_dict']["l2_regularization_strength"])

    loss_fn = SigmoidLoss()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))

    max_acc = 0
    print('Starting training at', datetime.today())
    for epoch in range(1, int(gnn_parameter['train_epoch_test']) + 1):
        start = time.time()
        train_loss = 0
        val_loss = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in graph_data.train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(graph_data.train_data_loader.dataset)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision, train_recall, train_int_ap, train_ap = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in graph_data.val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)

            val_loss /= len(graph_data.val_data_loader.dataset)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, val_precision, val_recall, val_int_ap, val_ap = do_compute_metrics(val_probas_pred, val_ground_truth)

            if val_acc>max_acc:
                max_acc = val_acc
                model_save(model, optimizer, graph_data.data_logger_save, num, fold)

        scheduler.step()
        print(f'Epoch: {epoch} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')

    print('Starting testing at', datetime.today())
    test_model = model_load(graph_data.data_logger_save, num, fold)

    if graph_data.transductive_flag:
        ### transductive
        test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap = eval_performance(graph_data.test_data_loader, test_model, device)
        print('==============================', fold, '==============================')
        print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
        print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')
        print('\n')
        return [test_acc, test_auc_roc, test_f1, test_precision, test_recall, test_int_ap, test_ap]
    else:
        ### inductive
        s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap = eval_performance(graph_data.val_data_loader, test_model, device)
        s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap = eval_performance(graph_data.test_data_loader, test_model, device)
        print('==============================', fold, '==============================')
        print(f'\t\ts1_acc: {s1_acc:.4f}, s1_auc_roc: {s1_auc_roc:.4f},s1_f1: {s1_f1:.4f},s1_precision:{s1_precision:.4f}')
        print(f'\t\ts1_recall: {s1_recall:.4f}, s1_int_ap: {s1_int_ap:.4f},s1_ap: {s1_ap:.4f}')
        print(f'\t\ts2_acc: {s2_acc:.4f}, s2_auc_roc: {s2_auc_roc:.4f},s2_f1: {s2_f1:.4f},s2_precision:{s2_precision:.4f}')
        print(f'\t\ts2_recall: {s2_recall:.4f}, s2_int_ap: {s2_int_ap:.4f},s2_ap: {s2_ap:.4f}')
        print('\n')
        return [[s1_acc, s1_auc_roc, s1_f1, s1_precision, s1_recall, s1_int_ap, s1_ap],[s2_acc, s2_auc_roc, s2_f1, s2_precision, s2_recall, s2_int_ap, s2_ap]]

