# Implementation Plan: GraphStorm Training Enhancements

## Overview

This implementation plan covers three main enhancements to GraphStorm:
1. Docker environment modernization (PyTorch 2.4.0, DGL 2.4.0, Ubuntu 22.04, CUDA cu124)
2. Max-steps parameter for training control
3. Edge weight support for contrastive loss in link prediction

The implementation follows an incremental approach, building and testing each component before integration.

## Tasks

- [x] 1. Set up feature branch and update Docker environment
  - Create new feature branch: `feature/training-enhancements`
  - Update `docker/sagemaker/Dockerfile.sm` with new dependency versions
  - Make `docker/build_docker_sagemaker.sh` executable
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 4.1_

- [ ]* 1.1 Write Docker environment verification tests
  - Create `tests/docker-tests/test_environment.py`
  - Test PyTorch, DGL, Ubuntu, CUDA, transformers, autogluon, einops versions
  - Test build script permissions
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8_

- [ ] 2. Implement max-steps parameter support
  - [x] 2.1 Add max_steps configuration parameter
    - Modify `python/graphstorm/config/argument.py`
    - Add `--max-steps` command line argument
    - Add `max_steps` property to `GSConfig` class
    - Add validation for positive integer values
    - _Requirements: 2.1_

  - [ ]* 2.2 Write property test for max-steps parameter acceptance
    - **Property 1: Max Steps Parameter Acceptance**
    - **Validates: Requirements 2.1, 2.5**
    - Create `tests/property-tests/test_max_steps_properties.py`
    - Use hypothesis to test parameter acceptance with random positive integers

  - [x] 2.3 Implement max-steps logic in link prediction trainer
    - Modify `python/graphstorm/trainer/lp_trainer.py`
    - Update `fit()` method to accept `max_steps` parameter
    - Add step counter check in training loop
    - Add early exit logic when max_steps reached
    - Add logging for max-steps termination
    - _Requirements: 2.2, 2.4_

  - [ ]* 2.4 Write property test for max-steps training termination
    - **Property 2: Max Steps Training Termination**
    - **Validates: Requirements 2.2, 2.4, 2.6**
    - Test that training stops at exactly max_steps
    - Test that logging occurs
    - Test that post-training logic executes

  - [ ]* 2.5 Write unit tests for max-steps functionality
    - Test backward compatibility (max_steps=None)
    - Test integration with link prediction task
    - Test error handling for invalid values
    - _Requirements: 2.3, 2.5_

- [x] 3. Checkpoint - Verify max-steps implementation
  - Build Docker image and test max-steps parameter
  - Run all max-steps tests
  - Ensure all tests pass, ask the user if questions arise

- [x] 4. Implement edge weight support for contrastive loss
  - [x] 4.1 Remove contrastive loss restriction in configuration
    - Modify `python/graphstorm/config/argument.py`
    - Update `lp_edge_weight_for_loss` property
    - Remove warning and early return for contrastive loss
    - _Requirements: 3.1, 3.2_

  - [x] 4.2 Update contrastive loss function to accept edge weights
    - Modify `python/graphstorm/model/loss_func.py`
    - Update `LinkPredictContrastiveLossFunc.forward()` signature
    - Add edge_weights parameter
    - Implement weight normalization and application
    - Add validation for edge weight values (no NaN, inf, negative)
    - _Requirements: 3.3, 3.4_

  - [ ]* 4.3 Write property test for edge weight application
    - **Property 3: Edge Weight Application with Contrastive Loss**
    - **Validates: Requirements 3.1, 3.2, 3.7**
    - Create `tests/property-tests/test_edge_weight_properties.py`
    - Test that edge weights are applied without warnings
    - Test with random batch sizes and weight values

  - [ ]* 4.4 Write property test for edge weight scaling correctness
    - **Property 4: Edge Weight Scaling Correctness**
    - **Validates: Requirements 3.3, 3.4**
    - Test that weighted loss scales proportionally
    - Test mathematical correctness (finite, positive)

  - [x] 4.5 Update contrastive decoders to pass edge weights
    - Modify `python/graphstorm/model/lp_gnn.py`
    - Update `LinkPredictContrastiveDotDecoder.forward()`
    - Update `LinkPredictContrastiveDistMultDecoder.forward()`
    - Update `LinkPredictContrastiveRotatEDecoder.forward()`
    - Update `LinkPredictContrastiveTransEDecoder.forward()`
    - Add edge_weights parameter to all decoder forward methods
    - Pass edge_weights to loss function
    - _Requirements: 3.1_

  - [x] 4.6 Update model forward pass to extract and pass edge weights
    - Modify model forward method in `python/graphstorm/model/lp_gnn.py`
    - Extract edge weights from pos_edge_feats based on config
    - Pass edge_weights to decoder
    - Add error handling for missing weight fields
    - _Requirements: 3.1, 3.7_

  - [ ]* 4.7 Write unit tests for edge weight integration
    - Test integration with dot_product decoder
    - Test integration with localjoint negative sampler
    - Test error handling (missing field, invalid values, shape mismatch)
    - _Requirements: 3.5, 3.6_

- [x] 5. Checkpoint - Verify edge weight implementation
  - Run all edge weight tests
  - Test with sample training configuration
  - Ensure all tests pass, ask the user if questions arise

- [x] 6. Integration and end-to-end testing
  - [x] 6.1 Wire all components together
    - Ensure max-steps parameter flows from config to trainer
    - Ensure edge weights flow from config through model to loss
    - Update any integration points
    - _Requirements: 2.1, 2.2, 3.1_

  - [ ]* 6.2 Write integration tests
    - Test max-steps with edge weights together
    - Test full training pipeline with new features
    - Test on sample heterogeneous graph data
    - _Requirements: 2.1, 2.2, 2.5, 3.1, 3.5, 3.6_

  - [x] 6.3 Update documentation and examples
    - Add max-steps parameter to training documentation
    - Add edge weight with contrastive loss example
    - Update YAML configuration examples
    - _Requirements: 2.1, 3.1_

- [-] 7. Final checkpoint and commit
  - Run full test suite (unit tests, property tests, integration tests)
  - Build and test Docker image
  - Commit all changes with descriptive message
  - Push to personal GitHub fork
  - Ensure all tests pass, ask the user if questions arise
  - _Requirements: 4.2, 4.3, 4.4_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties with minimum 100 iterations
- Unit tests validate specific examples and edge cases
- The implementation preserves backward compatibility - existing code works without changes
