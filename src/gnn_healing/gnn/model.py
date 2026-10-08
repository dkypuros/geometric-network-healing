"""Chapter 3: equivariant relational message passing.

Layer (Eq. 3.4, R-GCN form):
    h_v^{l+1} = sigma( W_0 h_v^l + sum_{r in R'} sum_{u in N_r(v)} (1/|N_r(v)|) W_r h_u^l )
where R' = R u R^T (forward and transposed relations, Chapter 6 needs the transpose).
Permutation equivariance (Eq. 3.1) holds because every operation is a per-node map or a
sum over neighbours; node ids never enter the computation.

Heads (invariant read-outs per node):
    forecast    x_hat_v(t)   in R^K   (Eq. 5.2, self-supervised)
    degradation d_hat_v(t)   in R     (Eq. 6.1, supervised from the twin's latent field)
    rca         z_v(t)       in R     (Eq. 6.2, softmax over V gives P(origin = v))
"""
from __future__ import annotations
import torch
from torch import nn
from torch_geometric.nn import RGCNConv
from ..twin.schema import NodeType, RELATIONS, K


class HealingGNN(nn.Module):
    def __init__(self, window: int, hidden: int = 64, layers: int = 3,
                 num_types: int = len(NodeType), num_relations: int = 2 * len(RELATIONS),
                 dropout: float = 0.1):
        super().__init__()
        self.type_emb = nn.Embedding(num_types, hidden)
        self.encoder = nn.Sequential(nn.Linear(window * K, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList([RGCNConv(hidden, hidden, num_relations) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head_forecast = nn.Linear(hidden, K)
        self.head_degradation = nn.Linear(hidden, 1)
        self.head_rca = nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_type, node_type):
        h = self.encoder(x) + self.type_emb(node_type)
        for conv, norm in zip(self.convs, self.norms):
            h = h + self.dropout(torch.relu(norm(conv(h, edge_index, edge_type))))  # residual
        return {
            "forecast": self.head_forecast(h),
            "degradation": self.head_degradation(h).squeeze(-1),
            "rca": self.head_rca(h).squeeze(-1),
            "embedding": h,
        }
