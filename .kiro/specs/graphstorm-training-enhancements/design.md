# Design Document: GraphStorm Training Enhancements

## Overview

This design specifies enhancements to GraphStorm for heterogeneous GNN RGCN training. The changes include:

1. **Docker Environment Modernization**: Update dependencies to newer versions (PyTorch 2.4.0, DGL 2.4.0, Ubuntu 22.04, CUDA cu124)
2. **Max Steps Training Control**: Add `--max-steps` parameter to stop training before completing full epochs
3. **Edge Weight Support for Contrastive Loss**: Enable edge weights with contrastive loss in link prediction

These enhancements support learning joint embedding spaces for heterogeneous graphs with nodes like queries, products, brands, and shoppers, where edge frequencies vary significantly (e.g., 1000 clicks vs 1 click).

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Environment                        │
│  (Dockerfile.sm, build_docker_sagemaker.sh)                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Configuration Layer                         │
│  (argument.py - GSConfig)                                    │
│  - Parses --max-steps parameter                             │
│  - Returns edge weights for contrastive loss                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Training Loop                              │
│  (lp_trainer.py - GSgnnLinkPredictionTrainer)               │
│  - Checks max_steps condition                               │
│  - Breaks training when limit reached                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Loss Computation                           │
│  (Contrastive Loss Function)                                │
│  - Applies edge weights to positive samples                 │
│  - Scales loss by weight values                             │
└─────────────────────────────────────────────────────────────┘
```

### Design Decisions

**Decision 1: Max Steps Implementation Location**
- Implement in the training loop (lp_trainer.py) rather than the dataloader
- Rationale: Training control logic belongs in the trainer, not data loading. This keeps concerns separated and makes the feature work across all trainer types.

**Decision 2: Edge Weight Application Strategy**
- Apply weights by scaling the positive sample loss contribution
- Rationale: Mathematically equivalent to duplicating positive edges W times, but computationally efficient. Avoids memory overhead of actual duplication.

**Decision 3: Docker Version Selection**
- Use PyTorch 2.4.0 and DGL 2.4.0 (latest stable versions)
- Use Ubuntu 22.04 (LTS with extended support)
- Use CUDA cu124 (matches PyTorch 2.4.0 compatibility)
- Rationale: These versions provide bug fixes, performance improvements, and extended support while maintaining compatibility.

## Components and Interfaces

### 1. Docker Configuration

**Files Modified:**
- `docker/sagemaker/Dockerfile.sm`
- `docker/build_docker_sagemaker.sh`

**Changes:**
```dockerfile
# Base image update
FROM pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker  # GPU
FROM pytorch-training:2.4.0-cpu-py311-ubuntu22.04-sagemaker        # CPU

# DGL version update
ARG DGL_VERSION=2.4.0
RUN pip3 install dgl==${DGL_VERSION}+cu124  # GPU
RUN pip3 install dgl==${DGL_VERSION}        # CPU

# New dependencies
RUN pip3 install \
    transformers==5.1.0 \
    autogluon.tabular[all]==1.4.0 \
    einops \
    ...
```

**Script Permissions:**
```bash
chmod +x docker/build_docker_sagemaker.sh
```

### 2. Configuration Layer

**File:** `python/graphstorm/config/argument.py`

**Interface Changes:**

```python
class GSConfig:
    def __init__(self, ...):
        self._max_steps = None  # New field
        
    @property
    def max_steps(self) -> Optional[int]:
        """Maximum number of training steps before stopping.
        
        Returns:
            int or None: Maximum steps, or None for full epoch training
        """
        return self._max_steps
    
    @property
    def lp_edge_weight_for_loss(self):
        """Edge feature field name for edge weight.
        
        Modified to support contrastive loss.
        """
        if hasattr(self, "_lp_edge_weight_for_loss"):
            assert self.task_type == BUILTIN_TASK_LINK_PREDICTION
            
            # REMOVED: Warning and early return for contrastive loss
            # Now allows edge weights with contrastive loss
            
            edge_weights = self._lp_edge_weight_for_loss
            # ... rest of implementation unchanged
```

**Command Line Argument:**
```python
parser.add_argument("--max-steps",
                    type=int,
                    default=None,
                    help="Maximum number of training steps. "
                         "Training stops after this many steps, "
                         "even if epoch is incomplete.")
```

### 3. Training Loop

**File:** `python/graphstorm/trainer/lp_trainer.py`

**Modified Method:** `GSgnnLinkPredictionTrainer.fit()`

**Interface:**
```python
def fit(self, train_loader, num_epochs, 
        max_steps=None,  # New parameter
        val_loader=None,
        test_loader=None,
        ...):
    """Fit function for link prediction.
    
    Parameters
    ----------
    max_steps : int, optional
        Maximum number of training steps. If provided, training stops
        after this many steps regardless of epoch completion.
    """
```

**Implementation Logic:**
```python
total_steps = 0
for epoch in range(num_epochs):
    for i, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_loader):
        total_steps += 1
        
        # ... training step logic ...
        
        # Check max_steps condition
        if max_steps is not None and total_steps >= max_steps:
            if get_rank() == 0:
                logging.info("Reached max_steps=%d, stopping training", max_steps)
            break  # Exit inner loop
    
    # Check if we broke due to max_steps
    if max_steps is not None and total_steps >= max_steps:
        break  # Exit outer loop
    
    # ... end of epoch logic ...
```

### 4. Contrastive Loss with Edge Weights

**File:** `python/graphstorm/model/loss_func.py` (or relevant loss implementation)

**Current Contrastive Loss:**
```python
class LinkPredictContrastiveLossFunc:
    def __init__(self, temperature):
        self.temperature = temperature
    
    def forward(self, pos_scores, neg_scores):
        # pos_scores: [batch_size]
        # neg_scores: [batch_size, num_negatives]
        
        # Concatenate positive and negative scores
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Softmax over all scores (positive is index 0)
        loss = -torch.log_softmax(scores, dim=1)[:, 0]
        
        return loss.mean()
```

**Modified with Edge Weights:**
```python
class LinkPredictContrastiveLossFunc:
    def forward(self, pos_scores, neg_scores, edge_weights=None):
        # pos_scores: [batch_size]
        # neg_scores: [batch_size, num_negatives]
        # edge_weights: [batch_size] or None
        
        scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        scores = scores / self.temperature
        loss = -torch.log_softmax(scores, dim=1)[:, 0]
        
        # Apply edge weights if provided
        if edge_weights is not None:
            # Normalize weights to prevent scale issues
            weights_normalized = edge_weights / edge_weights.mean()
            loss = loss * weights_normalized
        
        return loss.mean()
```

**Decoder Interface Update:**

Decoders must pass edge weights to loss function:

```python
class LinkPredictContrastiveDotDecoder:
    def forward(self, pos_graph, neg_graph, embeddings, edge_weights=None):
        # ... compute pos_scores and neg_scores ...
        
        # Pass edge_weights to loss function
        loss = self.loss_func(pos_scores, neg_scores, edge_weights)
        return loss
```

**Model Forward Pass:**

The model must extract and pass edge weights:

```python
def forward(self, blocks, pos_graph, neg_graph, 
            node_feats, edge_feats, pos_edge_feats, input_nodes):
    # ... compute embeddings ...
    
    # Extract edge weights if configured
    edge_weights = None
    if self.config.lp_edge_weight_for_loss is not None:
        weight_field = self.config.lp_edge_weight_for_loss
        # Extract from pos_edge_feats based on weight_field
        edge_weights = pos_edge_feats[weight_field]
    
    # Pass to decoder
    loss = self.decoder(pos_graph, neg_graph, embeddings, edge_weights)
    return loss
```

## Data Models

### Configuration Data

```python
@dataclass
class TrainingConfig:
    max_steps: Optional[int] = None
    lp_edge_weight_for_loss: Optional[Union[str, Dict[Tuple[str, str, str], List[str]]]] = None
    lp_loss_func: str = "contrastive"
    contrastive_loss_temperature: float = 0.1
```

### Edge Weight Data

```python
# Edge weights are stored as edge features in the graph
# Format: torch.Tensor of shape [num_edges]
# Values: Positive floats representing edge importance/frequency

# Example:
# For edge (query_1, engaged, product_1) with 1000 clicks: weight = 1000.0
# For edge (query_2, engaged, product_2) with 1 click: weight = 1.0
```

### Training State

```python
@dataclass
class TrainingState:
    current_epoch: int
    current_step: int
    total_steps: int
    max_steps_reached: bool
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Max Steps Parameter Acceptance

*For any* valid positive integer value provided as --max-steps, the configuration system should accept it without error and make it available to the training loop.

**Validates: Requirements 2.1, 2.5**

### Property 2: Max Steps Training Termination

*For any* training configuration with max-steps set to N, training should stop after exactly N steps, log the stopping reason, save model checkpoints, and complete evaluation logic.

**Validates: Requirements 2.2, 2.4, 2.6**

### Property 3: Edge Weight Application with Contrastive Loss

*For any* link prediction configuration with edge weights and contrastive loss, the system should apply edge weights to the loss calculation without logging warnings, and handle both positive and negative edges consistently.

**Validates: Requirements 3.1, 3.2, 3.7**

### Property 4: Edge Weight Scaling Correctness

*For any* positive edge with weight W, the contribution of that edge to the total loss should be approximately W times its contribution with weight 1.0, maintaining mathematical correctness of the loss function.

**Validates: Requirements 3.3, 3.4**

## Error Handling

### Docker Build Errors

**Scenario:** Base image not available or dependency version conflicts

**Handling:**
- Docker build will fail with clear error message indicating which dependency failed
- User should check AWS ECR availability for base images
- User should verify dependency compatibility matrix

**Example:**
```bash
# If PyTorch 2.4.0 image not available
ERROR: manifest for pytorch-training:2.4.0-gpu-py311-cu124-ubuntu22.04-sagemaker not found
```

### Max Steps Configuration Errors

**Scenario:** Invalid max-steps value (negative, zero, or non-integer)

**Handling:**
```python
if max_steps is not None:
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError(f"max_steps must be a positive integer, got {max_steps}")
```

**Scenario:** max-steps smaller than evaluation frequency

**Handling:**
- Training will stop at max-steps even if evaluation hasn't occurred
- Log warning: "max_steps={N} is smaller than eval_frequency={M}, evaluation may not occur"

### Edge Weight Errors

**Scenario:** Edge weight field not found in graph data

**Handling:**
```python
if weight_field not in pos_edge_feats:
    raise KeyError(f"Edge weight field '{weight_field}' not found in edge features. "
                   f"Available fields: {list(pos_edge_feats.keys())}")
```

**Scenario:** Edge weights contain invalid values (NaN, inf, negative)

**Handling:**
```python
if edge_weights is not None:
    if torch.isnan(edge_weights).any():
        raise ValueError("Edge weights contain NaN values")
    if torch.isinf(edge_weights).any():
        raise ValueError("Edge weights contain infinite values")
    if (edge_weights < 0).any():
        raise ValueError("Edge weights must be non-negative")
```

**Scenario:** Edge weight tensor shape mismatch

**Handling:**
```python
if edge_weights.shape[0] != pos_scores.shape[0]:
    raise ValueError(f"Edge weights shape {edge_weights.shape} does not match "
                     f"batch size {pos_scores.shape[0]}")
```

### Backward Compatibility

**Scenario:** Existing training scripts without max-steps parameter

**Handling:**
- Default value is None, preserving original behavior
- No code changes required for existing users

**Scenario:** Existing configurations with edge weights but not contrastive loss

**Handling:**
- No changes to existing behavior
- Edge weights continue to work with cross-entropy and BPR loss as before

## Testing Strategy

### Dual Testing Approach

This feature requires both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and integration points
- **Property tests**: Verify universal properties across randomized inputs

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing Configuration

**Library Selection:**
- Python: Use `hypothesis` library for property-based testing
- Configuration: Minimum 100 iterations per property test
- Each test must reference its design document property using the tag format:
  - `# Feature: graphstorm-training-enhancements, Property {N}: {property_text}`

### Test Categories

#### 1. Docker Environment Tests (Unit Tests)

**Test: Verify PyTorch Version**
```python
def test_pytorch_version():
    """Verify Docker image contains PyTorch 2.4.0"""
    # Build image and check: python -c "import torch; print(torch.__version__)"
    assert version == "2.4.0"
```

**Test: Verify DGL Version**
```python
def test_dgl_version():
    """Verify Docker image contains DGL 2.4.0"""
    assert version == "2.4.0"
```

**Test: Verify Ubuntu Version**
```python
def test_ubuntu_version():
    """Verify Docker image uses Ubuntu 22.04"""
    # Check: cat /etc/os-release
    assert "22.04" in version_info
```

**Test: Verify CUDA Version**
```python
def test_cuda_version():
    """Verify GPU Docker image uses CUDA cu124"""
    assert "12.4" in cuda_version
```

**Test: Verify New Dependencies**
```python
def test_new_dependencies():
    """Verify transformers 5.1.0, autogluon 1.4.0, and einops are installed"""
    assert transformers_version == "5.1.0"
    assert autogluon_version == "1.4.0"
    assert einops_installed
```

**Test: Verify Script Permissions**
```python
def test_build_script_executable():
    """Verify build_docker_sagemaker.sh has executable permissions"""
    import os
    import stat
    st = os.stat("docker/build_docker_sagemaker.sh")
    assert st.st_mode & stat.S_IXUSR  # User execute permission
```

#### 2. Max Steps Tests

**Property Test: Max Steps Parameter Acceptance**
```python
# Feature: graphstorm-training-enhancements, Property 1: Max Steps Parameter Acceptance
@given(max_steps=st.integers(min_value=1, max_value=10000))
def test_max_steps_parameter_acceptance(max_steps):
    """For any valid positive integer, config should accept max-steps parameter"""
    config = GSConfig(max_steps=max_steps)
    assert config.max_steps == max_steps
```

**Property Test: Max Steps Training Termination**
```python
# Feature: graphstorm-training-enhancements, Property 2: Max Steps Training Termination
@given(max_steps=st.integers(min_value=10, max_value=100))
def test_max_steps_training_termination(max_steps):
    """For any max-steps value, training should stop at exactly that step"""
    # Create mock trainer with small dataset
    trainer = create_mock_trainer()
    
    # Train with max_steps
    with capture_logs() as logs:
        trainer.fit(train_loader, num_epochs=10, max_steps=max_steps)
    
    # Verify stopped at correct step
    assert trainer.total_steps == max_steps
    
    # Verify logging
    assert f"Reached max_steps={max_steps}" in logs
    
    # Verify post-training logic executed
    assert trainer.checkpoints_saved
    assert trainer.evaluation_completed
```

**Unit Test: Backward Compatibility**
```python
def test_max_steps_none_preserves_behavior():
    """When max-steps is not provided, training should complete full epochs"""
    trainer = create_mock_trainer()
    trainer.fit(train_loader, num_epochs=2, max_steps=None)
    
    # Should complete both epochs
    assert trainer.completed_epochs == 2
```

**Unit Test: Max Steps with Link Prediction**
```python
def test_max_steps_with_link_prediction():
    """Max-steps should work with link prediction task on SageMaker"""
    config = create_link_prediction_config(max_steps=50)
    trainer = GSgnnLinkPredictionTrainer(config)
    
    # Should complete without errors
    trainer.fit(train_loader, num_epochs=5, max_steps=50)
    assert trainer.total_steps == 50
```

#### 3. Edge Weight with Contrastive Loss Tests

**Property Test: Edge Weight Application**
```python
# Feature: graphstorm-training-enhancements, Property 3: Edge Weight Application with Contrastive Loss
@given(
    batch_size=st.integers(min_value=4, max_value=64),
    num_negatives=st.integers(min_value=10, max_value=100),
    weights=st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=4, max_size=64)
)
def test_edge_weight_application_contrastive(batch_size, num_negatives, weights):
    """For any configuration with edge weights and contrastive loss, 
    system should apply weights without warnings"""
    
    # Ensure weights match batch size
    weights = weights[:batch_size]
    
    config = GSConfig(
        lp_loss_func="contrastive",
        lp_edge_weight_for_loss="weight",
        contrastive_loss_temperature=0.1
    )
    
    # Verify config returns edge weight field (not None)
    assert config.lp_edge_weight_for_loss is not None
    
    # Create mock data with edge weights
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size, num_negatives)
    edge_weights = torch.tensor(weights)
    
    # Compute loss
    with capture_logs() as logs:
        loss_func = LinkPredictContrastiveLossFunc(temperature=0.1)
        loss = loss_func(pos_scores, neg_scores, edge_weights)
    
    # Verify no warning about disabled edge weights
    assert "does not work with contrastive" not in logs
    assert "Disable edge weight" not in logs
    
    # Verify loss is valid
    assert torch.isfinite(loss)
    assert loss > 0
```

**Property Test: Edge Weight Scaling Correctness**
```python
# Feature: graphstorm-training-enhancements, Property 4: Edge Weight Scaling Correctness
@given(
    batch_size=st.integers(min_value=4, max_value=32),
    weight_multiplier=st.floats(min_value=1.5, max_value=10.0)
)
def test_edge_weight_scaling_correctness(batch_size, weight_multiplier):
    """For any weight W, loss contribution should be approximately W times the unweighted loss"""
    
    # Create identical batches
    pos_scores = torch.randn(batch_size)
    neg_scores = torch.randn(batch_size, 50)
    
    loss_func = LinkPredictContrastiveLossFunc(temperature=0.1)
    
    # Compute unweighted loss
    weights_ones = torch.ones(batch_size)
    loss_unweighted = loss_func(pos_scores, neg_scores, weights_ones)
    
    # Compute weighted loss
    weights_scaled = torch.ones(batch_size) * weight_multiplier
    loss_weighted = loss_func(pos_scores, neg_scores, weights_scaled)
    
    # Verify scaling relationship (within 5% tolerance due to normalization)
    expected_loss = loss_unweighted * weight_multiplier
    assert torch.isclose(loss_weighted, expected_loss, rtol=0.05)
    
    # Verify mathematical correctness
    assert torch.isfinite(loss_weighted)
    assert loss_weighted > 0
```

**Unit Test: Integration with Dot Product Decoder**
```python
def test_edge_weights_with_dot_product_decoder():
    """Edge weights should work with dot_product decoder"""
    config = create_config(
        lp_decoder_type="dot_product",
        lp_loss_func="contrastive",
        lp_edge_weight_for_loss="weight"
    )
    
    model = create_model(config)
    
    # Create mock batch with edge weights
    batch = create_mock_batch_with_weights()
    
    # Should compute loss without errors
    loss = model(batch)
    assert torch.isfinite(loss)
```

**Unit Test: Integration with LocalJoint Sampler**
```python
def test_edge_weights_with_localjoint_sampler():
    """Edge weights should work with localjoint negative sampler"""
    config = create_config(
        train_negative_sampler="localjoint",
        lp_loss_func="contrastive",
        lp_edge_weight_for_loss="weight"
    )
    
    # Create dataloader with localjoint sampler
    dataloader = create_dataloader(config)
    
    # Should sample and train without errors
    for batch in dataloader:
        loss = model(batch)
        assert torch.isfinite(loss)
```

#### 4. Error Handling Tests (Unit Tests)

**Test: Invalid Max Steps Values**
```python
def test_invalid_max_steps_values():
    """Should reject negative, zero, or non-integer max-steps"""
    with pytest.raises(ValueError):
        GSConfig(max_steps=-1)
    
    with pytest.raises(ValueError):
        GSConfig(max_steps=0)
    
    with pytest.raises(TypeError):
        GSConfig(max_steps="100")
```

**Test: Missing Edge Weight Field**
```python
def test_missing_edge_weight_field():
    """Should raise clear error when edge weight field not found"""
    config = GSConfig(lp_edge_weight_for_loss="missing_field")
    
    with pytest.raises(KeyError, match="Edge weight field 'missing_field' not found"):
        model(batch_without_weight_field)
```

**Test: Invalid Edge Weight Values**
```python
def test_invalid_edge_weight_values():
    """Should reject NaN, inf, or negative edge weights"""
    loss_func = LinkPredictContrastiveLossFunc(temperature=0.1)
    
    # Test NaN
    with pytest.raises(ValueError, match="NaN"):
        loss_func(pos_scores, neg_scores, torch.tensor([1.0, float('nan'), 2.0]))
    
    # Test inf
    with pytest.raises(ValueError, match="infinite"):
        loss_func(pos_scores, neg_scores, torch.tensor([1.0, float('inf'), 2.0]))
    
    # Test negative
    with pytest.raises(ValueError, match="non-negative"):
        loss_func(pos_scores, neg_scores, torch.tensor([1.0, -1.0, 2.0]))
```

### Test Execution

**Running Unit Tests:**
```bash
pytest tests/unit-tests/test_training_enhancements.py -v
```

**Running Property Tests:**
```bash
# Run with default 100 iterations
pytest tests/property-tests/test_training_properties.py -v

# Run with more iterations for thorough testing
pytest tests/property-tests/test_training_properties.py -v --hypothesis-iterations=1000
```

**Docker Tests:**
```bash
# Build image
./docker/build_docker_sagemaker.sh

# Run tests inside container
docker run graphstorm-sagemaker:latest python -m pytest tests/docker-tests/
```
