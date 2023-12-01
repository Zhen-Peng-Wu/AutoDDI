from torch_geometric.nn import GATConv,SAGEConv,GraphConv,GeneralConv,MFConv,LEConv
from torch_geometric.nn import TopKPooling,SAGPooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from autoddi.search_space.gcn_conv import GCNConv
from autoddi.search_space.pan_pool import PANPooling

def global_pooling_map(global_pooling_type):
    if global_pooling_type == 'global_max':
        global_pooling = global_max_pool
    elif global_pooling_type == 'global_mean':
        global_pooling = global_mean_pool
    elif global_pooling_type == 'global_add':
        global_pooling = global_add_pool
    else:
        raise Exception("Wrong global_pooling function")
    return global_pooling


def conv_map(conv_type, input_dim, hidden_dim, heads = 2):
    if conv_type == 'GCNConv':
        conv_layer = GCNConv(input_dim, hidden_dim)
    elif conv_type == 'GATConv':
        conv_layer = GATConv(input_dim, hidden_dim//heads, heads=heads)
    elif conv_type == 'SAGEConv':
        conv_layer = SAGEConv(input_dim, hidden_dim, normalize=True)
    elif conv_type == 'GraphConv':
        conv_layer = GraphConv(input_dim, hidden_dim)
    elif conv_type == 'GeneralConv':
        conv_layer = GeneralConv(input_dim, hidden_dim)
    elif conv_type == 'MFConv':
        conv_layer = MFConv(input_dim, hidden_dim)
    elif conv_type == 'LEConv':
        conv_layer = LEConv(input_dim, hidden_dim)
    else:
        raise Exception("Wrong conv function")
    return conv_layer


def bi_conv_map(bi_conv_type, input_dim, hidden_dim, heads = 2):
    if bi_conv_type == 'GCNConv':
        bi_conv_layer = GCNConv((input_dim,input_dim), hidden_dim)
    elif bi_conv_type == 'GATConv':
        bi_conv_layer = GATConv((input_dim,input_dim), hidden_dim//heads, heads=heads)
    elif bi_conv_type == 'SAGEConv':
        bi_conv_layer = SAGEConv((input_dim,input_dim), hidden_dim, normalize=True)
    elif bi_conv_type == 'GraphConv':
        bi_conv_layer = GraphConv((input_dim,input_dim), hidden_dim)
    elif bi_conv_type == 'GeneralConv':
        bi_conv_layer = GeneralConv((input_dim, input_dim), hidden_dim)
    elif bi_conv_type == 'MFConv':
        bi_conv_layer = MFConv((input_dim,input_dim), hidden_dim)
    elif bi_conv_type == 'LEConv':
        bi_conv_layer = LEConv((input_dim,input_dim), hidden_dim)
    else:
        raise Exception("Wrong bi_conv function")
    return bi_conv_layer


def local_pooling_map(local_pooling_type, hidden_dim):
    if local_pooling_type == 'TopKPool':
        local_pooling_layer = TopKPooling(hidden_dim, min_score=-1)
    elif local_pooling_type == 'SAGPool':
        local_pooling_layer = SAGPooling(hidden_dim, min_score=-1)
    elif local_pooling_type == 'PANPool':
        local_pooling_layer = PANPooling(hidden_dim, min_score=-1)
    else:
        raise Exception("Wrong local_pooling function")
    return local_pooling_layer
