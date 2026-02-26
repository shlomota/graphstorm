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
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch as th

from ..dataloading.dataloading import GSgnnEdgeDataLoader
from .gnn import GSgnnModel, GSgnnModelBase
from ..model.edge_decoder import LinkPredictionTestScoreInterface
from .loss_func import LinkPredictContrastiveLossFunc
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
            Node embedding normalization method.
        ground_ntype: str, optional
            Node type whose embeddings should be "grounded" to their input embeddings.
            Only applied when input and hidden dimensions match. Default: None.
        ground_method: str, optional
            Grounding method. ``"freeze"`` replaces the GNN output with the input embedding
            for ``ground_ntype``. ``"reconstruct"`` adds an MSE reconstruction loss that
            encourages the GNN output to stay close to the input embedding. Default: ``"freeze"``.
        ground_coef: float, optional
            Coefficient for the reconstruction loss when ``ground_method="reconstruct"``.
            Default: 0.1.
    """
    def __init__(self, alpha_l2norm, embed_norm_method=None,
                 ground_ntype=None, ground_method="freeze", ground_coef=0.1):
        super(GSgnnLinkPredictionModel, self).__init__()
        self.alpha_l2norm = alpha_l2norm
        self.embed_norm_method = embed_norm_method
        self.ground_ntype = ground_ntype
        self.ground_method = ground_method
        self.ground_coef = ground_coef
        self._ground_dim_warned = False  # Avoid repeated dimension-mismatch warnings

    def normalize_node_embs(self, embs):
        return normalize_node_embs(embs, self.embed_norm_method)

    # pylint: disable=unused-argument
    def forward(self, blocks, pos_graph,
        neg_graph, node_feats, edge_feats,
        pos_edge_feats=None, neg_edge_feats=None, input_nodes=None):
        """ The forward function for link prediction.

        .. versionchanged:: 0.4.0
            Add ``edge_feats`` into ``compute_embed_step`` in v0.4.0 to use edge features
            in message passing computation.
        """
        alpha_l2norm = self.alpha_l2norm
        recon_loss = th.tensor(0.)

        if blocks is None or len(blocks) == 0:
            # no GNN message passing, just compute node embeddings
            encode_embs = self.comput_input_embed(input_nodes, node_feats)
        elif self.ground_ntype is not None:
            # Grounding: use raw input features (e.g. pre-computed MiniLM embeddings)
            # directly, bypassing the learned input encoder projection.  This keeps
            # the saved embeddings in the same space as external queries so callers
            # can run MiniLM on a new query and search immediately without any
            # additional learned transform.
            gnn_embs = self.compute_embed_step(blocks, node_feats, input_nodes, edge_feats)

            if self.ground_ntype in gnn_embs and self.ground_ntype in node_feats:
                gnd_out = gnn_embs[self.ground_ntype]
                n_seeds = gnd_out.shape[0]
                # node_feats[ground_ntype] covers all input nodes; seed nodes are first.
                gnd_feat = node_feats[self.ground_ntype][:n_seeds]
                if gnd_out.shape[-1] != gnd_feat.shape[-1]:
                    if not self._ground_dim_warned:
                        logging.warning(
                            "Grounding skipped for '%s': feat dim %d != hidden dim %d. "
                            "Grounding only supported when dimensions match.",
                            self.ground_ntype, gnd_feat.shape[-1], gnd_out.shape[-1])
                        self._ground_dim_warned = True
                    encode_embs = gnn_embs
                elif self.ground_method == "freeze":
                    # Replace GNN output with raw input features for ground_ntype
                    encode_embs = dict(gnn_embs)
                    encode_embs[self.ground_ntype] = gnd_feat
                elif self.ground_method == "reconstruct":
                    # Use GNN output but add MSE loss pulling it toward raw features.
                    encode_embs = gnn_embs
                    recon_loss = th.nn.functional.mse_loss(
                        gnd_out, gnd_feat.detach())
                else:
                    raise ValueError(
                        f"Unknown ground_method '{self.ground_method}'. "
                        "Supported: 'freeze', 'reconstruct'.")
            else:
                encode_embs = gnn_embs
        else:
            # has GNN encoder, compute node embedding, or optional edge embeddings
            encode_embs = self.compute_embed_step(blocks, node_feats, input_nodes, edge_feats)

        # Call emb normalization.
        encode_embs = self.normalize_node_embs(encode_embs)

        # TODO add w_relation in calculating the score. The current is only valid for
        # homogenous graph.
        pos_score = self.decoder(pos_graph, encode_embs, pos_edge_feats)
        neg_score = self.decoder(neg_graph, encode_embs, neg_edge_feats)
        assert pos_score.keys() == neg_score.keys(), \
            "Positive scores and Negative scores must have edges of same" \
            f"edge types, but get {pos_score.keys()} and {neg_score.keys()}"

        # For contrastive loss, pass edge weights from pos_edge_feats to the loss function
        # so it can compute a weighted mean (approximating duplicated positive edges).
        if isinstance(self.loss_func, LinkPredictContrastiveLossFunc) \
                and pos_edge_feats is not None:
            edge_weights = {etype: feats for etype, feats in pos_edge_feats.items()}
            pred_loss = self.loss_func(pos_score, neg_score, edge_weights=edge_weights)
        else:
            pred_loss = self.loss_func(pos_score, neg_score)

        # add regularization loss to all parameters to avoid the unused parameter errors
        reg_loss = th.tensor(0.).to(pred_loss.device)
        # L2 regularization of dense parameters
        for d_para in self.get_dense_params():
            reg_loss += d_para.square().sum()

        # weighted addition to the total loss
        total_loss = pred_loss + alpha_l2norm * reg_loss
        if self.ground_ntype is not None and self.ground_method == "reconstruct":
            total_loss = total_loss + self.ground_coef * recon_loss.to(pred_loss.device)
        return total_loss

    def apply_ground_to_embeddings(self, emb, data, device, batch_size=1024):
        """ Post-process full-graph embeddings for the 'freeze' grounding method.

        Replaces the ``ground_ntype`` GNN embeddings with the raw input features
        (e.g. pre-computed MiniLM embeddings stored as 'feat') so that the saved
        embeddings live in the same space as external queries.  No additional
        distributed inference pass is required — the features are read directly
        from the distributed graph's node data.

        Parameters
        ----------
        emb : dict of Tensor
            GNN embeddings keyed by node type.
        data : GSgnnData
            The graph dataset.
        device : torch.device
            Unused; kept for API compatibility.
        batch_size : int
            Unused; kept for API compatibility.

        Returns
        -------
        dict of Tensor
            Updated embeddings with ``ground_ntype`` replaced by raw node features
            (if applicable).
        """
        if self.ground_ntype is None or self.ground_method != "freeze":
            return emb
        if self.ground_ntype not in emb:
            return emb

        g = data.g
        ntype = self.ground_ntype
        feat_field = data.node_feat_field
        if isinstance(feat_field, dict):
            feat_name = feat_field.get(ntype)
        else:
            feat_name = feat_field

        if feat_name is None or feat_name not in g.nodes[ntype].data:
            logging.warning(
                "Grounding skipped for '%s' during inference: "
                "raw feature '%s' not found in graph data.",
                ntype, feat_name)
            return emb

        gnd_inp = g.nodes[ntype].data[feat_name]  # DistTensor; device transfer on index
        gnd_out = emb[ntype]
        if gnd_inp.shape[-1] != gnd_out.shape[-1]:
            logging.warning(
                "Grounding skipped for '%s' during inference: "
                "feat dim %d != hidden dim %d.",
                ntype, gnd_inp.shape[-1], gnd_out.shape[-1])
            return emb

        emb = dict(emb)  # shallow copy to avoid mutating the original
        emb[ntype] = gnd_inp
        return emb

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
