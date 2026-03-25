# GraphStorm Training Enhancements

This document describes the new training enhancements added to GraphStorm.

## 1. Max Steps Parameter

### Overview
The `--max-steps` parameter allows you to stop training after a specified number of steps, regardless of epoch completion. This is useful for:
- Quick experimentation and debugging
- Time-constrained training runs
- Early stopping based on step count rather than epochs

### Usage

Add the `--max-steps` parameter to your training command:

```bash
python -m graphstorm.run.gsgnn_lp.gsgnn_lp \
    --max-steps 60 \
    --num-epochs 1 \
    --batch-size 16384 \
    ...
```

Or in your YAML configuration:

```yaml
gsf:
  basic:
    num_epochs: 1
    max_steps: 60
    batch_size: 16384
```

### Behavior
- Training will stop after exactly `max_steps` iterations
- Must be a positive integer
- If not specified, training runs for full epochs as usual
- Logging will indicate when max_steps limit is reached
- Post-training logic (evaluation, model saving) executes normally

### Example Log Output
```
INFO:root:Epoch 00000 | Batch 060 | Train Loss: 4.5752 | Time: 54.5699
INFO:root:Reached max_steps=60, stopping training
```

## 2. Node Embedding Grounding

### Overview
Node embedding grounding keeps certain node types (like "query") close to their input embeddings during training. This is useful when you want to preserve pre-trained features (e.g., text embeddings from BERT) without letting GNN layers modify them too much.

Two grounding methods are supported:
- **freeze**: Replace GNN output with input embeddings (bypass GNN layers completely)
- **reconstruct**: Add cosine similarity reconstruction loss to encourage embeddings to stay close to inputs

### Requirements
- Only works when input feature dimension matches hidden dimension
- If dimensions don't match, grounding is automatically disabled with a warning

### Usage

#### Method 1: Freeze (Bypass GNN)

This method completely bypasses GNN layers for the specified node type, using input embeddings directly:

```yaml
link_prediction:
  lp_loss_func: contrastive
  lp_decoder_type: dot_product
  node_embed_grounding_ntype: "query"
  node_embed_grounding_method: "freeze"  # Default
```

#### Method 2: Reconstruct (Add Cosine Similarity Loss)

This method adds a reconstruction loss term to keep embeddings close to inputs while still allowing some GNN transformation:

```yaml
link_prediction:
  lp_loss_func: contrastive
  lp_decoder_type: dot_product
  node_embed_grounding_ntype: "query"
  node_embed_grounding_method: "reconstruct"
  node_embed_grounding_lambda: 0.1  # Weight for reconstruction loss
```

### Behavior
- **freeze method**: Query embeddings = input embeddings (no GNN transformation)
- **reconstruct method**: Adds `lambda * (1 - cosine_similarity(output_emb, input_emb))` to the loss
  - Loss range: [0, 2] where 0 = perfect alignment, 1 = orthogonal, 2 = opposite
  - Cosine similarity is scale-invariant and better captures semantic similarity than MSE
- Dimension check: Automatically validates input_dim == hidden_dim
- Warning logged if dimensions don't match (grounding disabled)
- Works with all contrastive loss decoders

### Example Configuration

Complete example with query grounding:

```yaml
version: 1.0

gsf:
  basic:
    task_type: link_prediction
    model_encoder_type: rgcn
    hidden_size: 384  # Must match query input feature dimension
    num_layers: 2
    num_epochs: 1
    batch_size: 16384
    
  link_prediction:
    lp_decoder_type: dot_product
    lp_loss_func: contrastive
    contrastive_loss_temperature: 0.1
    lp_embed_normalizer: "l2_norm"
    num_negative_edges: 100
    train_negative_sampler: localjoint
    
    # Node embedding grounding
    node_embed_grounding_ntype: "query"
    node_embed_grounding_method: "freeze"  # or "reconstruct"
    node_embed_grounding_lambda: 0.1  # Only used with "reconstruct"
    
    target_etype: ["query,engaged,product"]
    train_etype:
      - "query,engaged,product"
      - "product,belongs,brand"
```

### Use Cases

1. **Preserve text embeddings**: Keep query text embeddings from BERT unchanged
2. **Prevent overfitting**: Stop GNN from overfitting on small query datasets
3. **Hybrid approach**: Use "reconstruct" to allow some adaptation while staying close to inputs

## 3. Edge Weights for Contrastive Loss

### Overview
Edge weights are now supported with contrastive loss in link prediction tasks. This allows you to:
- Weight positive edges by importance (e.g., click frequency, interaction strength)
- Reflect real-world edge significance in training
- Improve model quality on imbalanced edge data

### Usage

#### Step 1: Add edge weights to your graph data
Ensure your positive edges have a weight feature. For example, if tracking product clicks:
- Edge with 1000 clicks: weight = 1000.0
- Edge with 1 click: weight = 1.0

#### Step 2: Configure edge weights in your training config

```yaml
link_prediction:
  lp_loss_func: contrastive
  lp_decoder_type: dot_product
  lp_edge_weight_for_loss: "click_count"  # Global weight field name
  target_etype: ["query,engaged,product"]
```

Or for per-edge-type weights:

```yaml
link_prediction:
  lp_edge_weight_for_loss:
    - "query,engaged,product:click_count"
    - "user,purchased,item:purchase_amount"
```

### Behavior
- Edge weights are normalized per batch for numerical stability
- Weights must be positive, finite values (no NaN, inf, or negative)
- Higher weights increase the importance of those positive edges in the loss
- Works with all contrastive loss decoders: dot_product, DistMult, RotatE, TransE
- Compatible with localjoint and other negative samplers

### Example Configuration

Complete example for heterogeneous graph with edge weights:

```yaml
version: 1.0

gsf:
  basic:
    task_type: link_prediction
    model_encoder_type: rgcn
    hidden_size: 384
    num_layers: 2
    num_epochs: 1
    batch_size: 16384
    max_steps: 60  # New: stop after 60 steps
    
  link_prediction:
    lp_decoder_type: dot_product
    lp_loss_func: contrastive
    contrastive_loss_temperature: 0.1
    lp_embed_normalizer: "l2_norm"
    num_negative_edges: 100
    train_negative_sampler: localjoint
    lp_edge_weight_for_loss: "weight"  # New: use edge weights
    target_etype: ["query,engaged,product"]
    train_etype:
      - "query,engaged,product"
      - "product,belongs,brand"
```

## 4. Docker Environment Updates

The Docker environment has been updated to:
- PyTorch 2.4.0 (from 2.3.0)
- DGL 2.4.0 (from 2.3.0)
- Ubuntu 22.04 (from 20.04)
- CUDA cu124 (from cu121)
- Transformers 5.1.0 (from 4.28.1)
- Added: autogluon.tabular[all]==1.4.0
- Added: einops

### Building the Docker Image

```bash
cd docker
./build_docker_sagemaker.sh
```

## Backward Compatibility

All changes are backward compatible:
- Existing training configs work without modification
- `max_steps` defaults to None (full epoch training)
- Edge weights default to None (unweighted loss)
- Docker base images remain compatible

## Implementation Details

### Max Steps
- Implemented in `GSConfig.max_steps` property with validation
- Training loop checks step count after each batch
- Early exit triggers same post-training logic as epoch completion

### Edge Weights
- Configuration: `GSConfig.lp_edge_weight_for_loss` property
- Data loading: Weights loaded via `pos_graph_edge_feats` parameter
- Model forward: Weights extracted from `pos_edge_feats` based on config
- Loss function: Weights normalized and applied to positive edge loss
- Validation: Checks for NaN, inf, and negative values

## References

- Requirements: `.kiro/specs/graphstorm-training-enhancements/requirements.md`
- Design: `.kiro/specs/graphstorm-training-enhancements/design.md`
- Tasks: `.kiro/specs/graphstorm-training-enhancements/tasks.md`
