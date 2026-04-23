## Purpose

Attention kernels for LLM inference, progressing from naive baseline through FlashAttention optimization. This module provides O(N) memory complexity attention computation via online softmax algorithm.

---

## Requirements

### Requirement: Naive Attention Baseline

The system SHALL provide a baseline attention implementation for correctness verification and performance comparison.

#### Scenario: Q·K^T Computation
- **WHEN** computing attention scores
- **THEN** the system SHALL correctly compute Q·K^T matrix multiplication

#### Scenario: Softmax Normalization
- **WHEN** applying softmax to attention scores
- **THEN** the output SHALL satisfy (0,1) range, sum=1, and monotonicity invariants

#### Scenario: Output Computation
- **WHEN** computing final attention output
- **THEN** the system SHALL correctly compute softmax(Q·K^T/√d)·V

#### Scenario: Numerical Accuracy
- **WHEN** comparing against PyTorch reference
- **THEN** numerical error SHALL be within FP32 ≤1e-3, FP16 ≤1e-2 tolerance

#### Scenario: N×N Attention Matrix
- **WHEN** processing sequence length N
- **THEN** the system SHALL handle the full N×N attention matrix

---

### Requirement: Shared Memory Tiling

The system SHALL use shared memory tiling to reduce global memory access overhead.

#### Scenario: Matrix Tiling
- **WHEN** processing attention computation
- **THEN** matrices SHALL be partitioned into shared memory tiles

#### Scenario: Coalesced Memory Access
- **WHEN** loading data from global memory
- **THEN** access patterns SHALL be coalesced for optimal bandwidth

#### Scenario: Tile-Level Computation
- **WHEN** computing in shared memory
- **THEN** tile-level matrix operations SHALL be performed efficiently

#### Scenario: Bank Conflict Elimination
- **WHEN** accessing shared memory
- **THEN** bank conflicts SHALL be eliminated via padding

#### Scenario: Bandwidth Improvement
- **WHEN** comparing to naive implementation
- **THEN** bandwidth SHALL improve by ≥2x

---

### Requirement: FlashAttention Algorithm

The system SHALL implement FlashAttention with O(N) memory complexity using online softmax.

#### Scenario: Streaming Softmax Computation
- **WHEN** computing attention scores for sequence length N
- **THEN** the system SHALL compute max and exp-sum in streaming manner without storing the full attention matrix

#### Scenario: Single Kernel Launch
- **WHEN** executing FlashAttention forward pass
- **THEN** the system SHALL complete all computation in a single kernel launch

#### Scenario: Memory Efficiency
- **WHEN** processing sequences of any length N
- **THEN** memory usage SHALL be O(N), not O(N²)

#### Scenario: Avoid N×N Matrix
- **WHEN** running FlashAttention
- **THEN** the system SHALL NOT store the full N×N attention matrix

#### Scenario: Causal Masking
- **WHEN** is_causal parameter is true
- **THEN** output[i] SHALL depend only on input[j≤i]

#### Scenario: Numerical Equivalence
- **WHEN** comparing FlashAttention output to standard attention
- **THEN** numerical error SHALL be within tolerance (FP32 ≤1e-3, FP16 ≤1e-2)
