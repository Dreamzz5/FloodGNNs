import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from timm.layers import Mlp, DropPath
import random

class DenseFAGCNConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 0.1,
        dropout: float = 0.0,
        if_expand=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps
        self.dropout = dropout

        self.lin = Linear(in_channels, out_channels, bias=False)
        self.att_l = Linear(out_channels, 1, bias=False)
        self.att_r = Linear(out_channels, 1, bias=False)

        self.reset_parameters()
        self.if_expand = if_expand

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.att_l.reset_parameters()

    def forward(self, x, x_0, adj):
        x = self.lin(x)

        # Compute attention scores
        alpha_l = self.att_l(x)
        alpha_r = self.att_r(x)

        # Compute attention matrix
        alpha = alpha_l.unsqueeze(1) * alpha_r.unsqueeze(2)
        alpha = torch.tanh(alpha)

        # Apply adjacency mask and dropout
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, 0.0).squeeze()
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Apply attention and aggregate
        out = torch.matmul(alpha, x)

        if self.eps != 0.0:
            out = out + self.eps * x_0

        return out


class DenseGATConv(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        if_expand=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.if_expand = if_expand

        self.lin = Linear(
            in_channels, heads * out_channels, bias=False, weight_initializer="glorot"
        )
        self.att_src = Parameter(torch.empty(1, 1, heads, out_channels))
        self.att_dst = Parameter(torch.empty(1, 1, heads, out_channels))

        if bias:
            self.bias = Parameter(
                torch.empty(heads * out_channels if concat else out_channels)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, x0, adj):
        if adj.sum() == adj.size(0):
            return x
        x = x.unsqueeze(0) if x.dim() == 2 else x  # [B, N, F]
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj  # [B, N, N]

        H, C = self.heads, self.out_channels
        B, N, _ = x.size()
        x = self.lin(x).view(B, N, H, C)

        alpha = torch.sum(x * self.att_src, dim=-1).unsqueeze(1) + torch.sum(
            x * self.att_dst, dim=-1
        ).unsqueeze(2)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.masked_fill(adj.unsqueeze(-1) == 0, float("-inf")).softmax(dim=2)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.matmul(alpha.movedim(3, 1), x.movedim(2, 1)).movedim(1, 2)

        out = out.reshape(B, N, H * C) if self.concat else out.mean(dim=2)

        if self.bias is not None:
            out += self.bias
        if self.if_expand:
            out = torch.concat([out, x.squeeze(2)])
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, heads={self.heads})"


class AttentionLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_nodes,
        num_heads=1,
        qkv_bias=False,
        dropout=0.5,
        if_expand=False,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = in_channels // num_heads

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)

        self.out_proj = nn.Linear(in_channels, out_channels)
        self.pos_proj = nn.Linear(num_nodes, in_channels)

    def forward(self, x, x0, adj):
        if adj.sum() == adj.size(0):
            return x
        pos = self.pos_proj(adj)
        query, key, value = self.qkv(x).chunk(3, -1)
        qs = torch.stack(torch.split(query, self.head_dim, dim=-1), dim=-3)
        ks = torch.stack(torch.split(key, self.head_dim, dim=-1), dim=-3)
        vs = torch.stack(torch.split(value, self.head_dim, dim=-1), dim=-3)
        out = (
            torch.nn.functional.scaled_dot_product_attention(qs, ks, vs, attn_mask=adj)
            .transpose(-3, -2)
            .flatten(start_dim=-2)
        )
        x = self.out_proj(out * 0.1 + x)

        return x


class DenseGINConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, eps=0, train_eps=False, if_expand=False
    ):
        super(DenseGINConv, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.if_expand = if_expand
    def forward(self, x, x0, adj):
        # GIN不需要特殊的邻接矩阵预处理
        adj /= adj.sum(dim=-1, keepdims=True)
        out = torch.matmul(adj, x)
        out = out + (1 + self.eps) * x
        if self.if_expand:
            out = torch.concat([out, x])
        return self.nn(out)


class DenseSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, if_expand=False):
        super(DenseSAGEConv, self).__init__()
        self.lin_rel = nn.Linear(in_channels, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)
        self.if_expand = if_expand
    def forward(self, x, x0, adj):
        if self.if_expand: x, residual = x.chunk(2)
        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = self.lin_rel(out) + self.lin_root(x)
        if self.if_expand: out = torch.concat([out, residual])
        return out


class DenseChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K, if_expand=False):
        super(DenseChebConv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        self.if_expand = if_expand

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, x0, adj):
        if self.if_expand: x, redidual = x.chunk(2)
        # 计算规范化的拉普拉斯矩阵
        d = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        norm_adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)

        # 计算单位矩阵 I 和规范化邻接矩阵 L
        I = torch.eye(adj.size(-1), device=adj.device)
        L = I - norm_adj

        # 计算切比雪夫多项式
        Tx_0 = x
        Tx_1 = torch.matmul(L, x)
        out = torch.matmul(Tx_0, self.weight[0]) + torch.matmul(Tx_1, self.weight[1])

        if self.K > 2:
            for k in range(2, self.K):
                Tx_2 = 2 * torch.matmul(L, Tx_1) - Tx_0
                out += torch.matmul(Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        out += self.bias

        if self.if_expand:
            out = torch.concat([out, redidual])
        return out


class DenseGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, if_expand=False):
        super(DenseGCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.if_expand = if_expand
    def forward(self, x, x0, adj):
        if self.if_expand: x, redidual = x.chunk(2)
        adj.fill_diagonal_(2)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        support = self.linear(x)
        out = torch.matmul(adj, support)
        if self.if_expand:
            out = torch.concat([out, redidual])
        return out


class FloodGNN(nn.Module):
    def __init__(
        self, in_channels, out_channels, eps=0, order=1, train_eps=True, if_expand=False
    ):
        super(FloodGNN, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            # DropPath(0.5),
        )
        self.proj = nn.Linear(in_channels * order, in_channels)
        self.order = order
        self.if_expand = if_expand
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))

    def forward(self, x, x0, adj):
        # 计算度矩阵
        d = adj.sum(dim=-1)
        # 计算D^(-1/2)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        # 计算规范化邻接矩阵: D^(-1/2) * A * D^(-1/2)
        adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)
        x_aggregate = torch.matmul(adj, x)
        if self.if_expand:
            x_aggregate = torch.concat([x_aggregate, x])
        return self.nn(x_aggregate)


class DenseGCN2Conv(nn.Module):
    def __init__(self, c_in, c_out, alpha=0.1, theta=0.5, if_expand = False):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.alpha = alpha
        self.theta = theta
        self.linear = nn.Linear(c_in, c_out, bias=True)
        self.if_expand = if_expand
    def forward(self, x, x0, adj):
        adj.fill_diagonal_(2)
        # 计算度矩阵
        if self.if_expand: x, redidual = x.chunk(2)
        d = adj.sum(dim=-1)
        # 计算D^(-1/2)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        # 计算规范化邻接矩阵: D^(-1/2) * A * D^(-1/2)
        adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)

        x = (1 - self.alpha) * x + self.alpha * x0
        h = self.theta * self.linear(x) + (1 - self.theta) * x0
        out = torch.matmul(adj, h)
        if self.if_expand:
            out = torch.concat([out, redidual])
        return out


class AdaptiveGraphConv(nn.Module):
    def __init__(
        self, c_in, c_out, adj_matrix, alpha, conv_type="ChebNet", if_expand=False
    ):
        super().__init__()
        self.conv_type = conv_type
        self.register_buffer("adj_matrix", adj_matrix)

        conv_classes = {
            "ChebNet": lambda: DenseChebConv(c_in, c_out, K=3, if_expand=if_expand),
            "GAT": lambda: DenseGATConv(c_in, c_out, if_expand=if_expand),
            "GraphSAGE": lambda: DenseSAGEConv(c_in, c_out, if_expand=if_expand),
            "GCNII": lambda: DenseGCN2Conv(c_in, c_out, if_expand=if_expand),
            "GCN": lambda: DenseGCNConv(c_in, c_out, if_expand=if_expand),
            "GIN": lambda: DenseGINConv(c_in, c_out, if_expand=if_expand),
            # "Identity": lambda: lambda a, b: a,
            "FAGCN": lambda: DenseFAGCNConv(c_in, c_out, if_expand=if_expand),
            "Transformer": lambda: AttentionLayer(
                c_in, c_out, adj_matrix.shape[0], if_expand=if_expand
            ),
        }

        self.conv = conv_classes.get(
            conv_type, lambda: ValueError(f"Unsupported convolution type: {conv_type}")
        )()

    def forward(self, x, x0=None):
        # x shape: (batch_size, c_in, num_nodes, seq_len)
        batch_size, c_in, num_nodes, seq_len = x.shape
        x = x.permute(0, 3, 2, 1)  # (batch_size, seq_len, num_nodes, c_in)
        x = x.reshape(
            -1, num_nodes, x.size(-1)
        )  # (batch_size * seq_len, num_nodes, c_in)

        if x0 is not None:
            x0 = x0.permute(0, 3, 2, 1).reshape(-1, num_nodes, x0.size(1))
        x = self.conv(x, x0, self.adj_matrix)

        x = x.view(-1, seq_len, num_nodes, c_in)
        return x.permute(0, 3, 2, 1)  # (batch_size, c_out, num_nodes, seq_len)


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        adj=None,
        conv_type="GCN",
        if_expand=False,
        alpha=0.1,
    ):
        super().__init__()
        self.conv_type = conv_type
        # edge_index, edge_weight = (
        #     self.prepare_graph_data(adj) if adj is not None else (None, None)
        # )
        self.graph_conv = AdaptiveGraphConv(
            input_dim,
            input_dim,
            adj,
            conv_type=conv_type,
            alpha=alpha,
            if_expand=if_expand,
        )
        self.if_expand = if_expand
        self.dropout = nn.Dropout(0.15)
        self.act = nn.ReLU()

    def forward(self, x, x0):
        h = self.graph_conv(x, x0)
        return h + x

    @staticmethod
    def prepare_graph_data(adj):
        edge_index = adj.nonzero().t().contiguous()
        edge_weight = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_weight


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class GCN_Point(nn.Module):
    """
    Paper: Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting
    Link: https://arxiv.org/abs/2208.05233
    Official Code: https://github.com/zezhishao/STID
    """

    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self.num_nodes = model_args["num_nodes"]
        self.node_dim = model_args["node_dim"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.embed_dim = model_args["embed_dim"]
        self.output_len = model_args["output_len"]
        self.num_layer = model_args["num_layer"]
        self.temp_dim_tid = model_args["temp_dim_tid"]
        self.temp_dim_diw = model_args["temp_dim_diw"]
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]

        self.if_time_in_day = model_args["if_T_i_D"]
        self.if_day_in_week = model_args["if_D_i_W"]
        self.if_spatial = model_args["if_node"]
        self.adj = model_args["adj_mx"]
        self.conv_type = model_args["conv_type"]
        self.if_expand = model_args["expand"]
        self.alpha = model_args.get("alpha", 0.1)

        # spatial embeddings
        if self.if_spatial:
            node_emb = torch.zeros(2, self.num_nodes)
            # node_emb[0] = 5; node_emb[1] = -5
            self.node_emb = nn.Parameter(node_emb)
            self.droppath = DropPath(0.8)
            # nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        # encoding
        self.hidden_dim = (
            self.embed_dim
            # + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_day_in_week)
            + self.temp_dim_diw * int(self.if_time_in_day)
        )
        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    self.hidden_dim,
                    self.hidden_dim,
                    adj=self.adj,
                    conv_type=self.conv_type,
                    if_expand=self.if_expand,
                    alpha=self.alpha,
                )
                for _ in range(self.num_layer)
            ]
        )

        # # regression
        num_layers = 2 if self.if_expand else 1
        in_channels = self.hidden_dim * self.input_len * num_layers
        if self.if_expand == False: 
            self.regression_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.output_len,
                kernel_size=(1, 1),
                bias=True,
            )
        else:
            # in_channels += self.node_dim
            self.regression_layer = nn.ModuleList(
                [
                    nn.Conv2d(
                        in_channels // 2,
                        self.output_len,
                        kernel_size=(1, 1),
                        bias=True,
                    )
                    for i in range(num_layers)
                ]
            )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Feed forward of STID.

        Args:
            history_data (torch.Tensor): history data with shape [B, L, N, C]

        Returns:
            torch.Tensor: prediction with shape [B, L, N, C]
        """

        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., 1]
            # In the datasets used in STID, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
            # If you use other datasets, you may need to change this line.
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)
            ]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., 2]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)
            ]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        time_series_emb = self.time_series_emb_layer(input_data.transpose(1, 3))
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)
        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # print(node_emb[0].shape, time_series_emb.shape)
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + tem_emb, dim=1)
        # encoding
        x0 = hidden
        if self.if_expand:
            hidden = hidden.repeat(2, 1, 1, 1)
        for layer in self.encoder:
            hidden = layer(hidden, x0)

        if self.if_expand:
            # hidden = torch.cat(hidden.chunk(2), 1)
            # hidden = hidden.transpose(2, 3).flatten(1, 2)
            # hidden = torch.concat([hidden] + node_emb, 1)
            
            # hidden = torch.concat([hidden, node_emb[0]], dim=1).unsqueeze(-1)
            # prediction = self.regression_layer(hidden)

            # probs = self.softmax_layer(self.node_emb)
            hidden = hidden.transpose(2, 3).flatten(1, 2).unsqueeze(-1)
            hidden_graph, hidden_node = hidden.chunk(2)
            # print(hidden_graph.shape, self.regression_layer[0])
            prediction_graph = self.regression_layer[0](hidden_graph)
            prediction_node = self.regression_layer[0](hidden_node)

            # # Replace argmax with softmax for differentiability
            choice_probs = torch.nn.functional.softmax(node_emb, dim=1)
            prediction = (choice_probs[:, 0].unsqueeze(1) * prediction_node +
                        choice_probs[:, 1].unsqueeze(1) * prediction_graph)
            if self.training:
                prediction = torch.concat([prediction, prediction_node, prediction_graph])
            
            # For debugging, you can still print the distribution of choices
            # print(f"Choice distribution: {torch.unique(torch.argmax(choice_probs, dim=1), return_counts=True)}")
        else:
            hidden = hidden.transpose(2, 3).flatten(1, 2)
            prediction = self.regression_layer(hidden.unsqueeze(-1))

        return prediction

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # 从Gumbel(0,1)分布中采样
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / tau  # 重参数化技巧
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # 重参数化技巧
        ret = y_soft
    return ret