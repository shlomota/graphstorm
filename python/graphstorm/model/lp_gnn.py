"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GNN model for link prediction in GraphStorm.
"""
import abc
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch as th

from ..dataloading.dataloading import GSgnnEdgeDataLoader
from .gnn import GSgnnModel, GSgnnModelBase
from ..model.edge_decoder import LinkPredictionTestScoreInterface
from .utils import normalize_node_embs
from ..eval.utils import calc_ranking

class GSgnnLinkPredictionModelInterface:
    """ The interface for GraphStorm link prediction model.

    This interface defines one method: ``forward()`` for training. Link prediction models
    should inherite this interface and implement this method.
    """
    @abc.abstractmethod
    def forward(self, blocks, pos_graph, neg_graph,
        node_feats, edge_feats, pos_edge_feats=None, neg_edge_feats=None, input_nodes=None):
        """ The forward function for link prediction.

        This method is used for training. It takes a list of DGL message flow graphs (MFGs),
        node features, and edge features of a mini-batch as inputs, and
        computes the loss of the model in the mini-batch as the return value. More
        detailed information about DGL MFG can be found in `DGL Neighbor Sampling
        Overview
        <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.

        Parameters
        ----------
        blocks: list of DGL MFGs
            Sampled subgraph in the list of DGL message flow graph (MFG) format. More
            detailed information about DGL MFG can be found in `DGL Neighbor Sampling
            Overview
            <https://docs.dgl.ai/stochastic_training/neighbor_sampling_overview.html>`_.
        pos_graph : a DGLGraph
            The graph that contains the positive edges.
        neg_graph : a DGLGraph
            The graph that contains the negative edges.
        node_feats : dict of Tensors
            The input node features of the message passing graph.
        edge_feats : dict of Tensors
            The input edge features of the message passing graph.
        input_nodes: dict of Tensors
            The input nodes of a mini-batch.

        Returns
        -------
        float: The loss of prediction of this mini-batch.
        """

# pylint: disable=abstract-method
class GSgnnLinkPredictionModelBase(GSgnnModelBase, GSgnnLinkPredictionModelInterface):
    """ GraphStorm GNN model base class for link-prediction tasks.

    This base class extends GraphStorm ``GSgnnModelBase`` and
    ``GSgnnLinkPredictionModelInterface``. When users want to define a customized link
    prediction GNN model and train the model in GraphStorm, the model class needs to
    inherit from this base class, and implement the required methods including ``forward()``,
    ``predict()``, ``save_model()``, ``restore_model()`` and ``create_optimizer()``.
    """

    def normalize_node_embs(self, embs):
        """ By default do nothing.

            One can implement his/her own node normalization method or call
            .utils.normalize_node_embs to leverage the builtin normalization
            methods.
        """
        return embs

class GSgnnLinkPredictionModel(GSgnnModel, GSgnnLinkPredictionModelInterface):
    """ GraphStorm GNN model for link prediction

        Parameters
        ----------
        alpha_l2norm : float
            The alpha for L2 normalization.
        embed_norm_method: str
            Node embedding normalization method
        node_embed_grounding_ntype: str, optional
            Node type to ground (keep close to input embeddings). Default: None.
        node_embed_grounding_method: str, optional
            Grounding method: "freeze" (bypass GNN) or "reconstruct" (add MSE loss).
            Default: "freeze".
        node_embed_grounding_lambda: float, optional
            Weight for reconstruction loss when using "reconstruct" method. Default: 0.1.
    """
    def __init__(self, alpha_l2norm, embed_norm_method=None,
                 node_embed_grounding_ntype=None,
                 node_embed_grounding_method="freeze",
                 node_embed_grounding_lambda=0.1):
        super(GSgnnLinkPredictionModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm
        self.embed_norm_method = embed_norm_method
        self.node_embed_grounding_ntype = node_embed_grounding_ntype
        self.node_embed_grounding_method = node_embed_grounding_method
        self.node_embed_grounding_lambda = node_embed_grounding_lambda

    def normalize_node_embs(self, embs):
        return normalize_node_embs(embs, self.embed_norm_method)

    # pylint: disable=unused-argument
    def forward(self, blocks, pos_graph,
        neg_graph, node_feats, edge_feats,
        pos_edge_feats=None, neg_edge_feats=None, input_nodes=None,
        edge_weight_field=None):
        """ The forward function for link prediction.

        .. versionchanged:: 0.4.0
            Add ``edge_feats`` into ``compute_embed_step`` in v0.4.0 to use edge features
            in message passing computation.
        
        .. versionchanged:: 0.5.0
            Add ``edge_weight_field`` parameter to support edge weights with contrastive loss.
            Add node embedding grounding support with "freeze" and "reconstruct" methods.
        """
        alpha_l2norm = self.alpha_l2norm
        reconstruction_loss = th.tensor(0.)
        
        if blocks is None or len(blocks) == 0:
            # no GNN message passing, just compute node embeddings
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        else:
            # Compute input embeddings first
            input_embs = self.comput_input_embed(input_nodes, node_feats)
            
            # has GNN encoder, compute node embedding, or optional edge embeddings
            gnn_embs = self.compute_embed_step(blocks, node_feats, input_nodes, edge_feats)
            
            # Apply node embedding grounding if configured
            if self.node_embed_grounding_ntype is not None:
                grounding_ntype = self.node_embed_grounding_ntype
                
                # Check if grounding is applicable
                if grounding_ntype in gnn_embs and grounding_ntype in input_embs:
                    input_dim = input_embs[grounding_ntype].shape[-1]
                    output_dim = gnn_embs[grounding_ntype].shape[-1]
                    
                    if input_dim != output_dim:
                        # Dimension mismatch - log warning and skip grounding
                        import logging
                        logging.warning(
                            f"Node embedding grounding skipped for '{grounding_ntype}': "
                            f"input dimension ({input_dim}) != hidden dimension ({output_dim}). "
                            f"Grounding only works when input and hidden dimensions match."
                        )
                        encode_embs = gnn_embs
                    else:
                        # Dimensions match - apply grounding
                        if self.node_embed_grounding_method == "freeze":
                            # Method 1: Replace GNN output with input embeddings (bypass GNN)
                            encode_embs = {}
                            for ntype in gnn_embs:
                                if ntype == grounding_ntype:
                                    encode_embs[ntype] = input_embs[ntype]
                                else:
                                    encode_embs[ntype] = gnn_embs[ntype]
                        elif self.node_embed_grounding_method == "reconstruct":
                            # Method 2: Add reconstruction loss
                            encode_embs = gnn_embs
                            query_output = gnn_embs[grounding_ntype]
                            query_input = input_embs[grounding_ntype]
                            reconstruction_loss = th.nn.functional.mse_loss(query_output, query_input)
                        else:
                            # Unknown method - use GNN embeddings as-is
                            import logging
                            logging.warning(
                                f"Unknown grounding method '{self.node_embed_grounding_method}'. "
                                f"Valid options: 'freeze', 'reconstruct'. Using GNN embeddings."
                            )
                            encode_embs = gnn_embs
                else:
                    # Grounding node type not found in embeddings
                    encode_embs = gnn_embs
            else:
                # No grounding configured
                encode_embs = gnn_embs

        # Call emb normalization.
        encode_embs = self.normalize_node_embs(encode_embs)

        # TODO add w_relation in calculating the score. The current is only valid for
        # homogenous graph.
        pos_score = self.decoder(pos_graph, encode_embs, pos_edge_feats)
        neg_score = self.decoder(neg_graph, encode_embs, neg_edge_feats)
        assert pos_score.keys() == neg_score.keys(), \
            "Positive scores and Negative scores must have edges of same" \
            f"edge types, but get {pos_score.keys()} and {neg_score.keys()}"
        
        # Extract edge weights if configured and available
        edge_weights = None
        if edge_weight_field is not None and pos_edge_feats is not None:
            edge_weights = {}
            if isinstance(edge_weight_field, str):
                # Global weight field name
                for etype in pos_score.keys():
                    if etype in pos_edge_feats and edge_weight_field in pos_edge_feats[etype]:
                        edge_weights[etype] = pos_edge_feats[etype][edge_weight_field]
            elif isinstance(edge_weight_field, dict):
                # Per-edge-type weight field names
                for etype, field_names in edge_weight_field.items():
                    if etype in pos_edge_feats and field_names[0] in pos_edge_feats[etype]:
                        edge_weights[etype] = pos_edge_feats[etype][field_names[0]]
            
            # Only pass edge_weights if we found any
            if len(edge_weights) == 0:
                edge_weights = None
        
        pred_loss = self.loss_func(pos_score, neg_score, edge_weights)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # Ensure reconstruction_loss is on the same device
        if not reconstruction_loss.is_cuda and pred_loss.is_cuda:
            reconstruction_loss = reconstruction_loss.to(pred_loss.device)

        # weighted addition to the total loss
        total_loss = pred_loss + alpha_l2norm * reg_loss
        
        # Add reconstruction loss if using "reconstruct" method
        if self.node_embed_grounding_method == "reconstruct" and reconstruction_loss.item() > 0:
            total_loss = total_loss + self.node_embed_grounding_lambda * reconstruction_loss
        
        return total_loss

def lp_mini_batch_predict(model, emb, loader, device, return_batch_lengths=False):
    """ Perform mini-batch prediction for link prediction and return rankings
        of true edges in the predicted scores.

        This function follows full-graph GNN embedding inference.
        After having the GNN embeddings, we need to perform mini-batch
        computation to make predictions on the GNN embeddings.

        Note: callers should call model.eval() before calling this function
        and call model.train() after when doing training.

        Parameters
        ----------
        model : GSgnnModel
            The GraphStorm GNN model
        emb : dict of Tensor
            The GNN embeddings
        loader : GSgnnEdgeDataLoader
            The GraphStorm dataloader
        device: th.device
            Device used to compute test scores
        return_batch_lengths: bool, default False
            Whether to return the lengths of each batch of edges for each ranking value.

        Returns
        -------
        rankings: dict[str, torch.Tensor], if `return_batch_lengths` was False
            Rankings of positive scores in format of {etype: ranking}
        rankings, batch_lengths: tuple[dict, dict], if `return_batch_lengths` was True
            A tuple of rankings of positive scores in format of {etype: ranking},
            and the corresponding batch lengths for each ranking value.
    """
    decoder = model.decoder
    return run_lp_mini_batch_predict(decoder,
                                     emb,
                                     loader,
                                     device,
                                     return_batch_lengths)

def run_lp_mini_batch_predict(
        decoder,
        emb: Dict[str, th.Tensor],
        loader: GSgnnEdgeDataLoader,
        device: Union[th.device, int],
        return_batch_lengths=False,
    ):
    """ Perform mini-batch link prediction with the given decoder.

        This function follows full-graph GNN embedding inference.
        After having the GNN embeddings, we need to perform mini-batch
        computation to make predictions on the GNN embeddings.

        Note: callers should call model.eval() before calling this function
        and call model.train() after when doing training.

        Parameters
        ----------
        decoder : LinkPredictNoParamDecoder or LinkPredictLearnableDecoder
            The GraphStorm link prediction decoder model
        emb : dict of Tensor
            The GNN embeddings
        loader : GSgnnEdgeDataLoader
            The GraphStorm dataloader
        device: th.device or int
            Device used to compute test scores
        return_batch_lengths: bool, default False
            Whether to return the candidate list sizes of each ranking value.

        Returns
        -------
        rankings: dict[tuple, torch.Tensor], if `return_batch_lengths` was False
            Rankings of positive scores in format of {etype: ranking}
        rankings, batch_lengths: tuple[dict, dict], if `return_batch_lengths` was True
            A tuple of rankings of positive scores in format of {etype: ranking},
            and the corresponding batch lengths for each ranking value.
    """
    with th.no_grad():
        ranking: Dict[Tuple, List[th.Tensor]] = defaultdict(list)
        batch_lengths: Dict[Tuple, List[th.Tensor]] = defaultdict(list)
        assert isinstance(decoder, LinkPredictionTestScoreInterface), \
            f"The decoder must implement LinkPredictionTestScoreInterface, got {decoder=}"
        for pos_neg_tuple, neg_sample_type in loader:
            score = \
                decoder.calc_test_scores(
                    emb, pos_neg_tuple, neg_sample_type, device)
            for canonical_etype, s in score.items():
                # We do not concatenate rankings into a single
                # ranking tensor to avoid unnecessary data copy.
                pos_score, neg_score = s
                assert pos_score.shape[0] == neg_score.shape[0], \
                    "There should be as many negative lists as there are positive examples"
                score_ranking = calc_ranking(pos_score, neg_score)
                ranking[canonical_etype].append(score_ranking)
                # Set the number of candidates for each positive example
                # (equal to neg_score.shape[0])
                # which will be the number of negatives (equal to neg_score.shape[1])
                # plus one for the positive example
                lengths_tensor = th.tensor(neg_score.shape[0] * [neg_score.shape[1] + 1])
                # Ensure rankings and lengths are on the same device
                lengths_tensor = lengths_tensor.to(score_ranking.device)

                batch_lengths[canonical_etype].append(lengths_tensor)

        rankings: Dict[Tuple, th.Tensor] = {}
        batch_length_tensors: Dict[Tuple, th.Tensor] = {}
        for canonical_etype, rank in ranking.items():
            rankings[canonical_etype] = th.cat(rank, dim=0)
            etype_lengths = batch_lengths[canonical_etype]
            batch_length_tensors[canonical_etype] = th.cat(etype_lengths, dim=0)

    if return_batch_lengths:
        return rankings, batch_length_tensors
    return rankings
