# Requirements Document

## Introduction

This document specifies requirements for enhancements to GraphStorm, an enterprise graph ML framework, to support heterogeneous GNN RGCN training for learning joint embedding spaces. The enhancements include Docker dependency updates, training control improvements, and edge weight support for contrastive loss in link prediction tasks.

## Glossary

- **GraphStorm**: Enterprise graph machine learning framework for training GNN models
- **RGCN**: Relational Graph Convolutional Network, a type of heterogeneous GNN
- **Link_Prediction_Task**: Training task that predicts connections between nodes in a graph
- **Contrastive_Loss**: Loss function that learns embeddings by contrasting positive and negative examples
- **Edge_Weight**: Numerical value representing the importance or frequency of a connection between nodes
- **SageMaker**: AWS service for training and deploying machine learning models
- **Training_Step**: Single forward and backward pass through a batch of training data
- **Epoch**: Complete pass through the entire training dataset
- **DGL**: Deep Graph Library, framework for graph neural networks
- **Docker_Image**: Containerized environment for running GraphStorm training

## Requirements

### Requirement 1: Docker Environment Modernization

**User Story:** As a machine learning engineer, I want to use updated versions of PyTorch, DGL, and system dependencies, so that I can leverage performance improvements and bug fixes in newer releases.

#### Acceptance Criteria

1. THE Docker_Image SHALL use PyTorch version 2.4.0
2. THE Docker_Image SHALL use DGL version 2.4.0
3. THE Docker_Image SHALL use Ubuntu 22.04 as the base image
4. THE Docker_Image SHALL use CUDA cu124 for GPU support
5. THE Docker_Image SHALL include transformers version 5.1.0
6. THE Docker_Image SHALL include autogluon.tabular[all] version 1.4.0
7. THE Docker_Image SHALL include the einops package
8. THE build_docker_sagemaker.sh script SHALL have executable permissions

### Requirement 2: Training Duration Control

**User Story:** As a machine learning engineer, I want to stop training after a specific number of steps, so that I can control training duration precisely in multi-instance SageMaker environments without waiting for full epochs.

#### Acceptance Criteria

1. WHEN the --max-steps parameter is provided, THE GraphStorm SHALL accept it as a valid training configuration parameter
2. WHEN training reaches the step count specified by --max-steps, THE GraphStorm SHALL stop training and proceed to post-training logic
3. WHEN --max-steps is not provided, THE GraphStorm SHALL continue training for full epochs as before
4. WHEN training stops due to --max-steps, THE GraphStorm SHALL log the stopping reason clearly
5. THE --max-steps parameter SHALL work with Link_Prediction_Task on SageMaker
6. WHEN training stops at max-steps, THE GraphStorm SHALL save model checkpoints and complete evaluation

### Requirement 3: Edge Weight Support for Contrastive Loss

**User Story:** As a machine learning engineer, I want to use edge weights with contrastive loss in link prediction, so that I can reflect the importance and frequency of different connections in my training data (e.g., 1000 clicks vs 1 click).

#### Acceptance Criteria

1. WHEN lp_edge_weight_for_loss is configured with contrastive loss, THE GraphStorm SHALL apply edge weights to the loss calculation
2. WHEN edge weights are applied to contrastive loss, THE GraphStorm SHALL NOT log a warning about edge weights being disabled
3. WHEN a positive edge has weight W, THE Contrastive_Loss SHALL approximate the effect of having W duplicate positive edges in the training data
4. THE edge weight implementation SHALL maintain mathematical correctness of the contrastive loss function
5. THE edge weight implementation SHALL work with the dot_product decoder type
6. THE edge weight implementation SHALL work with the localjoint negative sampler
7. WHEN edge weights are applied, THE GraphStorm SHALL handle both positive and negative edges consistently

### Requirement 4: Version Control Integration

**User Story:** As a developer, I want all changes committed to a feature branch and pushed to a personal fork, so that I can maintain clean version control and potentially contribute back to the official GraphStorm repository.

#### Acceptance Criteria

1. THE implementation SHALL be committed to a new feature branch
2. THE feature branch SHALL contain all changes from Requirements 1, 2, and 3
3. THE commits SHALL have clear, descriptive commit messages
4. THE changes SHALL be pushed to a personal GitHub fork
5. THE branch SHALL be structured to allow future merging to the official GraphStorm repository
