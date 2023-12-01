# data_name = 'drugbank_inductive'
# save_suffix = '_search'
# gnn_parameter = {}
# gnn_parameter['opt_type_dict'] = {"learning_rate": 1e-3, "l2_regularization_strength": 5e-4}
# gnn_parameter['train_epoch_test'] = 200
# gnn_parameter['train_epoch'] = 5
# gnn_parameter['mode'] = 'test' # 'test' or 'search'

data_name = 'drugbank_transductive'
save_suffix = '_search'
gnn_parameter = {}
gnn_parameter['opt_type_dict'] = {"learning_rate": 0.01, "l2_regularization_strength": 5e-4}
gnn_parameter['train_epoch_test'] = 200
gnn_parameter['train_epoch'] = 5
gnn_parameter['mode'] = 'test' # 'test' or 'search'

# data_name = 'twosides_transductive'
# save_suffix = '_search'
# gnn_parameter = {}
# gnn_parameter['opt_type_dict'] = {"learning_rate": 0.01, "l2_regularization_strength": 5e-4}
# gnn_parameter['train_epoch_test'] = 120
# gnn_parameter['train_epoch'] = 1
# gnn_parameter['mode'] = 'test' # 'test' or 'search'

search_parameter = {}
search_parameter['device'] = 'cuda:0'
search_parameter['controller_train_epoch'] = 100
search_parameter['search_scale'] = 100
search_parameter['test_gnn_num'] = 5
search_parameter['controller_lr'] = 3.5e-4
search_parameter['cuda'] = True
search_parameter['entropy_coeff']= 1e-4
search_parameter['ema_baseline_decay'] = 0.95
search_parameter['discount'] = 1.0
search_parameter['controller_train_parallel_num'] = 1
search_parameter['controller_grad_clip'] = 0.0
search_parameter['tanh_c'] = 2.5
search_parameter['softmax_temperature'] = 5.0
