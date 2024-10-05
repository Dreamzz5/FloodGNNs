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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

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
        self, in_channels, out_channels, eps=0, train_eps=False
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

    def forward(self, x, x0, adj):
        adj /= adj.sum(dim=-1, keepdims=True)
        out = torch.matmul(adj, x)
        out = out + (1 + self.eps) * x
        return self.nn(out)


class DenseSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseSAGEConv, self).__init__()
        self.lin_rel = nn.Linear(in_channels, out_channels)
        self.lin_root = nn.Linear(in_channels, out_channels)

    def forward(self, x, x0, adj):
        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        out = self.lin_rel(out) + self.lin_root(x)
        return out


class DenseChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super(DenseChebConv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, x0, adj):
        d = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        norm_adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)

        I = torch.eye(adj.size(-1), device=adj.device)
        L = I - norm_adj

        Tx_0 = x
        Tx_1 = torch.matmul(L, x)
        out = torch.matmul(Tx_0, self.weight[0]) + torch.matmul(Tx_1, self.weight[1])

        if self.K > 2:
            for k in range(2, self.K):
                Tx_2 = 2 * torch.matmul(L, Tx_1) - Tx_0
                out += torch.matmul(Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2

        out += self.bias

        return out


class DenseGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseGCNConv, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x, x0, adj):
        adj.fill_diagonal_(2)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return out



class DenseGCN2Conv(nn.Module):
    def __init__(self, c_in, c_out, alpha=0.1, theta=0.5):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.alpha = alpha
        self.theta = theta
        self.linear = nn.Linear(c_in, c_out, bias=True)

    def forward(self, x, x0, adj):
        adj.fill_diagonal_(2)
        d = adj.sum(dim=-1)
        d_inv_sqrt = torch.pow(d, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        adj = d_inv_sqrt.unsqueeze(-1) * adj * d_inv_sqrt.unsqueeze(-2)

        x = (1 - self.alpha) * x + self.alpha * x0
        h = self.theta * self.linear(x) + (1 - self.theta) * x0
        out = torch.matmul(adj, h)
        return out


class AdaptiveGraphConv(nn.Module):
    def __init__(
        self, c_in, c_out, adj_matrix, alpha, conv_type="ChebNet"
    ):
        super().__init__()
        self.conv_type = conv_type
        self.register_buffer("adj_matrix", adj_matrix)

        conv_classes = {
            "ChebNet": lambda: DenseChebConv(c_in, c_out, K=3),
            "GAT": lambda: DenseGATConv(c_in, c_out),
            "GraphSAGE": lambda: DenseSAGEConv(c_in, c_out),
            "GCNII": lambda: DenseGCN2Conv(c_in, c_out, alpha=alpha),
            "GCN": lambda: DenseGCNConv(c_in, c_out),
            "GIN": lambda: DenseGINConv(c_in, c_out),
            "FAGCN": lambda: DenseFAGCNConv(c_in, c_out),
            "Transformer": lambda: AttentionLayer(
                c_in, c_out, adj_matrix.shape[0]
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
        alpha=0.1,
    ):
        super().__init__()
        self.conv_type = conv_type
        self.graph_conv = AdaptiveGraphConv(
            input_dim,
            input_dim,
            adj,
            conv_type=conv_type,
            alpha=alpha,
        )
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



class FloodGNN(nn.Module):
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
        self.time_of_day_size = model_args["time_of_day_size"]
        self.day_of_week_size = model_args["day_of_week_size"]
        self.adj = model_args["adj_mx"]
        self.conv_type = model_args["conv_type"]
        self.alpha = model_args.get("alpha", 0.1)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        # encoding
        self.hidden_dim = self.embed_dim
        self.encoder = nn.Sequential(
            *[
                MultiLayerPerceptron(
                    self.hidden_dim,
                    self.hidden_dim,
                    adj=self.adj,
                    conv_type=self.conv_type,
                    alpha=self.alpha,
                )
                for _ in range(self.num_layer)
            ]
        )

        # regression
        in_channels = self.hidden_dim * self.input_len
        self.regression_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.output_len,
                kernel_size=(1, 1),
                bias=True,
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

        # time series embedding
        time_series_emb = self.time_series_emb_layer(input_data.transpose(1, 3))
        # concatenate all embeddings
        hidden = torch.cat([time_series_emb], dim=1)
        # encoding
        x0 = hidden
        for layer in self.encoder:
            hidden = layer(hidden, x0)

        hidden = hidden.transpose(2, 3).flatten(1, 2)
        prediction = self.regression_layer(hidden.unsqueeze(-1))

        return prediction