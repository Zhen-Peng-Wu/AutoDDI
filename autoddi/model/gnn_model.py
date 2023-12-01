import torch
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm
from torch import nn
from autoddi.search_space.utils import conv_map, bi_conv_map, local_pooling_map, global_pooling_map

class GnnModel(torch.nn.Module):
    """
    Constructing the stack gcn model based on stack gcn architecture,
    realizing the stack gcn model forward process.

    Args:
        architecture: list
            the stack gcn architecture describe
        in_features: int
            the original input dimension for the stack gcn model

    Returns:
        output: tensor
            the output of the stack gcn model.
    """
    def __init__(self,
                 architecture,
                 in_features,
                 num_labels,
                 rel_total,
                 data_name,
                 hidden_dim=128):

        super(GnnModel, self).__init__()

        self.cells_num = architecture[0]
        self.architecture = architecture[1:]
        self.in_features = in_features
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.rel_total = rel_total

        self.initial_norm = LayerNorm(self.in_features)
        self.cells = []
        self.net_norms = torch.nn.ModuleList()
        
        input_dim = self.in_features
        for i in range(self.cells_num):
            cell = AutoDDI_Cell(self.architecture, input_dim, self.hidden_dim)
            self.add_module(f"cell{i}", cell)
            self.cells.append(cell)
            self.net_norms.append(LayerNorm(self.hidden_dim))
            input_dim = self.hidden_dim

        if data_name.find('drugbank') > -1:
            self.post_processing = Post_Pro(self.rel_total, self.hidden_dim, self.cells_num, self.num_labels)
        elif data_name.find('twosides') > -1:
            self.post_processing = Post_Pro_twosides(self.rel_total, self.hidden_dim, self.cells_num, self.num_labels)

        
    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)
        repr_h = []
        repr_t = []

        for i, cell in enumerate(self.cells):
            out = cell(h_data,t_data,b_graph)

            h_data = out[0]
            t_data = out[1]
            r_h = out[2]
            r_t = out[3]
            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)
        scores = self.post_processing(repr_h, repr_t, rels)
        return scores


class AutoDDI_Cell(nn.Module):
    def __init__(self, architecture, in_features, hidden_dim, n_heads=2):
        super().__init__()
        self.architecture = architecture
        self.in_features = in_features
        self.hidden_dim = hidden_dim

        ### dim /2 is because of: the feature of conv + the feature of bi_conv
        self.conv = conv_map(architecture[0], in_features, hidden_dim // 2, n_heads)
        self.bi_conv = bi_conv_map(architecture[1], in_features, hidden_dim // 2, n_heads)

        self.local_pooling = local_pooling_map(architecture[2], hidden_dim)
        self.global_pooling = global_pooling_map(architecture[3])

    def forward(self, h_data, t_data, b_graph):
        ## conv
        h_Rep = self.conv(F.elu(h_data.x), h_data.edge_index)
        t_Rep = self.conv(F.elu(t_data.x), t_data.edge_index)

        ## bi_conv
        t_biRep = self.bi_conv((F.elu(h_data.x), F.elu(t_data.x)), b_graph.edge_index)
        h_biRep = self.bi_conv((F.elu(t_data.x), F.elu(h_data.x)), b_graph.edge_index[[1, 0]])

        # add conv and bi_conv
        h_data.x = torch.cat([h_Rep, h_biRep], 1)
        t_data.x = torch.cat([t_Rep, t_biRep], 1)

        # local_pooling
        h_att_x, att_edge_index, att_edge_attr, h_att_batch = self.local_pooling(h_data.x, h_data.edge_index, batch=h_data.batch)[:4]
        t_att_x, att_edge_index, att_edge_attr, t_att_batch = self.local_pooling(t_data.x, t_data.edge_index, batch=t_data.batch)[:4]

        # global_pooling
        h_global_graph_emb = self.global_pooling(h_att_x, h_att_batch)
        t_global_graph_emb = self.global_pooling(t_att_x, t_att_batch)

        return h_data, t_data, h_global_graph_emb, t_global_graph_emb


class Post_Pro(nn.Module):
    def __init__(self, rel_total, hidden_dim, cells_num, num_labels):
        super().__init__()
        self.rel_total = rel_total
        self.hidden_dim = hidden_dim
        self.cells_num = cells_num
        self.num_labels = num_labels

        self.rel_emb = nn.Embedding(rel_total, hidden_dim * hidden_dim)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(cells_num * cells_num, num_labels)
        )

    def forward(self, heads, tails, rels):
        ### same as previous works, but they use coattention, we just use mlp
        rels = self.rel_emb(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        rels = rels.view(-1, self.hidden_dim, self.hidden_dim)
        scores = heads @ rels @ tails.transpose(-2, -1)

        scores = scores.view(-1, self.cells_num * self.cells_num)
        scores = self.mlp(scores)
        return scores


class Post_Pro_twosides(nn.Module):
    def __init__(self, rel_total, hidden_dim, cells_num, num_labels):
        super().__init__()
        self.rel_total = rel_total
        self.hidden_dim = hidden_dim
        self.cells_num = cells_num
        self.num_labels = num_labels

        self.rel_emb = nn.Embedding(rel_total, hidden_dim * 2)
        self.rel_proj = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(cells_num * cells_num, num_labels)
        )

    def forward(self, heads, tails, rels):
        ### same as previous works, but they use coattention, we just use mlp
        rels = self.rel_emb(rels)
        rels = self.rel_proj(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        pair = (heads.unsqueeze(-3) * tails.unsqueeze(-2)).unsqueeze(-2)
        rels = rels.view(-1, 1, 1, self.hidden_dim, 1)
        scores = ((torch.matmul(pair, rels)).squeeze(-1)).squeeze(-1)

        scores = scores.view(-1, self.cells_num * self.cells_num)
        scores = self.mlp(scores)

        return scores