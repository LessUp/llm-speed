## MODIFIED Requirements

### Requirement: Profiling and Verification
The system SHALL provide profiling tools and correctness verification for kernel development, and closeout verification MUST distinguish prepared-environment defects from environment setup failures.

#### Scenario: Nsight Compute Integration
- **WHEN** profiling kernels
- **THEN** Nsight Compute integration SHALL be available for detailed analysis

#### Scenario: Performance Metrics
- **WHEN** running benchmarks
- **THEN** TFLOPS, bandwidth, and SM occupancy SHALL be reported

#### Scenario: Correctness Reference
- **WHEN** verifying correctness
- **THEN** PyTorch SHALL be used as reference implementation

#### Scenario: Benchmark Comparison
- **WHEN** evaluating performance
- **THEN** comparison against cuBLAS/cuDNN SHALL be provided

#### Scenario: Bottleneck Identification
- **WHEN** analyzing performance
- **THEN** compute-bound vs memory-bound bottlenecks SHALL be identified

#### Scenario: Environment Baseline Failure
- **WHEN** verification cannot start because required local dependencies are absent
- **THEN** the project SHALL treat the result as an environment preparation issue that must be resolved before code-level verification is judged
