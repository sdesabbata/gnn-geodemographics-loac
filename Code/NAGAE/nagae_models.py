# Based on
# https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py [MIT license]
# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/argva_node_clustering.py [MIT license]

from typing import Optional, Tuple
import warnings

import torch
from torch import Tensor
from torch.nn import Module, Linear, BatchNorm1d
from torch.nn import functional as F
from torch_geometric.nn import GATConv, InnerProductDecoder
from torch_geometric.nn.inits import reset
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-15


# -------------------------- #
#   NAGAE model definition   #
# -------------------------- #

# This model is based on the Graph Auto-Encoder model from PyTorch Geometric
# which is based on the module described in the paper "Variational Graph Auto-Encoders"
# https://arxiv.org/abs/1611.07308

class NAGAE(torch.nn.Module):
    def __init__(
            self,
            encoder: Module,
            attributes_decoder: Module,
            edges_decoder: Optional[Module] = None
    ):
        super().__init__()
        self.encoder = encoder
        self.attributes_decoder = attributes_decoder
        self.edges_decoder = InnerProductDecoder() if edges_decoder is None else edges_decoder
        NAGAE.reset_parameters(self)

    def reset_parameters(self):
        # Resets all learnable parameters of the module
        reset(self.encoder)
        reset(self.attributes_decoder)
        reset(self.edges_decoder)

    def forward(self, attributes, edge_index) -> Tensor:
        # forward pass: encode, decode and return loss
        encoded = self.encode(attributes, edge_index)
        decoded = self.decode(encoded)
        return self.loss(decoded, attributes)

    def encode(self, attributes, edge_index) -> Tensor:
        # Runs the encoder and computes node-wise latent variables
        return self.encoder(attributes, edge_index)

    def decode(self, attributes) -> Tensor:
        # Runs the decoder and computes edge probabilities
        return self.attributes_decoder(attributes)

    def loss(self, reconstructed: Tensor, original: Tensor) -> Tensor:
        # Computes the Mean Square Error (MSE) for the attribute reconstructed by the decoder
        return F.mse_loss(reconstructed, original, reduction='mean')

    # The following functions are used for final testing after trainig
    # including attribute and edge reconstruction

    def __add_pos_neg(self, graph):
        # This function adds positive and negative sample edges to a graph,
        # which are used to compute the edge loss in the final test
        neg_edge_index = negative_sampling(
            edge_index=graph.edge_index,
            num_nodes=graph.x.size(0),
            num_neg_samples=graph.edge_index.size(1),
            method='sparse'
        )

        graph[f'pos_edge_label'] = torch.ones(graph.edge_index.size(1))
        graph[f'pos_edge_label_index'] = graph.edge_index

        graph[f'neg_edge_label'] = torch.zeros(graph.edge_index.size(1))
        graph[f'neg_edge_label_index'] = neg_edge_index

        return graph

    def test(self, test_graph) -> Tuple[float, float, float]:
        # Computes area under the ROC curve (AUC) and average precision (AP)
        # scores for the edges and the MSE (Mean Square Error) score for the attributes.

        # Add positive and negative sample edges
        test_graph = self.__add_pos_neg(test_graph)
        # Encode graph
        encoded = self.encode(test_graph.x, test_graph.edge_index)
        # Decode graph
        decoded = self.decode(encoded)

        # Attribute loss
        mse = self.loss(decoded, test_graph.x)

        # Attempt to use encoding to reconstruct edges
        auc = None
        ap = None
        try:
            pos_y = encoded.new_ones(test_graph.pos_edge_label_index.size(1))
            neg_y = encoded.new_zeros(test_graph.neg_edge_label_index.size(1))
            y = torch.cat([pos_y, neg_y], dim=0)

            pos_pred = self.edges_decoder(encoded, test_graph.pos_edge_label_index, sigmoid=True)
            neg_pred = self.edges_decoder(encoded, test_graph.neg_edge_label_index, sigmoid=True)
            pred = torch.cat([pos_pred, neg_pred], dim=0)

            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

            auc = roc_auc_score(y, pred)
            ap = average_precision_score(y, pred)
        except IndexError:
            warnings.warn("\n\nIndexError: error while testing positive and negative edges, ignoring testing batch edge reconstruction...\n\n")

        return float(mse), float(auc), float(ap)


# ---------------------- #
#   GAT-based encoder    #
# ---------------------- #

class GATEncoder(torch.nn.Module):
    def __init__(self,
            layers_features_in, layers_features_prep, layers_features_gatc, layers_features_post,
            num_gat_heads, negative_slope, dropout
        ):
        super(GATEncoder, self).__init__()

        self.negative_slope = negative_slope

        # --- Prep layers ---
        if len(layers_features_prep) == 0:
            self.sequential_prep = None
        else:
            self.sequential_prep = torch.nn.Sequential()
            self.sequential_prep.append(torch.nn.Linear(layers_features_in, layers_features_prep[0]))
            self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
            for i in range(len(layers_features_prep) - 1):
                self.sequential_prep.append(torch.nn.Linear(layers_features_prep[i], layers_features_prep[i + 1]))
                self.sequential_prep.append(torch.nn.LeakyReLU(negative_slope=negative_slope))

        # --- GAT layers ---
        self.gatconvs = torch.nn.ModuleList()
        # GAT layers except last
        for layers_features_gatc_h in layers_features_gatc[:-1]:
            self.gatconvs.append(GATConv(
                -1, layers_features_gatc_h,
                heads=num_gat_heads, negative_slope=negative_slope, dropout=dropout
            ))
        # Last GAT layer, averaged instead of concatenated
        self.gatconvs.append(GATConv(
            -1, layers_features_gatc[-1],
            heads=num_gat_heads, concat=False, negative_slope=negative_slope, dropout=dropout
        ))

        # --- Final layers ---
        self.sequential_post = torch.nn.Sequential()
        self.sequential_post.append(torch.nn.Linear(layers_features_gatc[-1], layers_features_post[0]))
        self.sequential_post.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
        for i in range(len(layers_features_post) - 1):
            self.sequential_post.append(torch.nn.Linear(layers_features_post[i], layers_features_post[i + 1]))

        self.final_batch_norm_1d = torch.nn.BatchNorm1d(layers_features_post[-1])

    def forward(self, attributes, edge_index):
        if self.sequential_prep is not None:
            attributes = self.sequential_prep(attributes)
        for a_gatconv in self.gatconvs:
            attributes = a_gatconv(attributes, edge_index)
        attributes = self.sequential_post(attributes)
        attributes = self.final_batch_norm_1d(attributes)
        return attributes


# -------------------------- #
#   Simple linear decoder    #
# -------------------------- #

class AttributeDecoder(torch.nn.Module):
    def __init__(self, layers_features, negative_slope):
        super().__init__()

        self.linear_decoder = torch.nn.Sequential()
        for i in range(len(layers_features) - 2):
            self.linear_decoder.append(torch.nn.Linear(layers_features[i], layers_features[i + 1]))
            self.linear_decoder.append(torch.nn.LeakyReLU(negative_slope=negative_slope))
        self.linear_decoder.append(torch.nn.Linear(layers_features[-2], layers_features[-1]))

    def forward(self, attributes):
        return self.linear_decoder(attributes)
